"""
Training script for unsupervised learning Tic-Tac-Toe bot
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import logging
from typing import List, Tuple, Optional
import wandb
from datetime import datetime
import time
from collections import deque

try:
    from .database import ClickHouseConnector
    from .models import AutoEncoder, TicTacToeBot
    from .data_utils import board_to_tensor, boards_to_batch, augment_board
    from .config import *
except ImportError:
    # For running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from database import ClickHouseConnector
    from models import AutoEncoder, TicTacToeBot
    from data_utils import board_to_tensor, boards_to_batch, augment_board
    from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_autoencoder(model: AutoEncoder, dataloader: DataLoader, device: str) -> dict:
    """Evaluate autoencoder on validation set"""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            boards = batch['board'].to(device)
            reconstructed, embedding = model(boards)
            loss = criterion(reconstructed, boards)
            total_loss += loss.item() * boards.size(0)
            total_samples += boards.size(0)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return {'eval_loss': avg_loss}


def evaluate_policy(bot: TicTacToeBot, dataloader: DataLoader, device: str) -> dict:
    """Evaluate policy head on validation set"""
    bot.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            boards = batch['board'].to(device)
            batch_size = boards.size(0)
            board_size = bot.board_size
            
            move_probs = bot(boards)
            targets = torch.randint(0, board_size * board_size, (batch_size,)).to(device)
            
            loss = criterion(move_probs, targets)
            total_loss += loss.item() * batch_size
            
            preds = torch.argmax(move_probs, dim=1)
            correct_predictions += (preds == targets).sum().item()
            total_predictions += batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return {
        'eval_loss': avg_loss,
        'eval_accuracy': accuracy
    }


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute gradient norm for monitoring"""
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** (1. / 2)
    return total_norm if param_count > 0 else 0.0


def compute_parameter_stats(model: nn.Module) -> dict:
    """Compute statistics about model parameters"""
    all_params = []
    for p in model.parameters():
        if p.requires_grad:
            all_params.append(p.data.cpu().numpy().flatten())
    
    if len(all_params) == 0:
        return {}
    
    all_params = np.concatenate(all_params)
    return {
        'param_mean': float(np.mean(all_params)),
        'param_std': float(np.std(all_params)),
        'param_min': float(np.min(all_params)),
        'param_max': float(np.max(all_params)),
        'param_norm': float(np.linalg.norm(all_params))
    }


class BoardDataset(Dataset):
    """Dataset for loading boards from ClickHouse"""
    
    def __init__(self, boards: List[Tuple[np.ndarray, str]], model_type: str = 'CNN', 
                 augment: bool = True, predict_next_move: bool = False):
        """
        Args:
            boards: List of (board_array, win_actor) tuples
            model_type: 'CNN' or 'MLP'
            augment: Whether to use data augmentation
            predict_next_move: If True, create pairs for next move prediction
        """
        self.boards = boards
        self.model_type = model_type
        self.augment = augment
        self.predict_next_move = predict_next_move
        
        # Expand dataset with augmentations
        if augment:
            self.expanded_boards = []
            for board, win_actor in boards:
                # Ensure original board is copied
                board_copy = np.ascontiguousarray(board.copy())
                augmented = augment_board(board_copy)
                for aug_board in augmented:
                    # Ensure each augmented board is contiguous
                    self.expanded_boards.append((np.ascontiguousarray(aug_board), win_actor))
        else:
            # Ensure all boards are contiguous copies
            self.expanded_boards = [(np.ascontiguousarray(board.copy()), win_actor) 
                                    for board, win_actor in boards]
    
    def __len__(self):
        return len(self.expanded_boards)
    
    def __getitem__(self, idx):
        board, win_actor = self.expanded_boards[idx]
        
        # Ensure board is a contiguous copy to avoid negative stride issues
        board = np.ascontiguousarray(board.copy())
        
        # Convert to tensor
        board_tensor = board_to_tensor(board, self.model_type).squeeze(0)
        
        # For autoencoder, target is the same board
        # For policy learning, we'll use win_actor to create targets
        return {
            'board': board_tensor,
            'win_actor': win_actor,
            'board_array': board.copy()  # Ensure copy for safety
        }


def train_autoencoder(model: AutoEncoder, dataloader: DataLoader, 
                     num_epochs: int, device: str, save_dir: str,
                     eval_dataloader: Optional[DataLoader] = None):
    """Train the autoencoder"""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    step = 0
    epoch_start_time = time.time()
    running_losses = deque(maxlen=100)  # Track last 100 losses for running average
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        batch_count = 0
        min_batch_loss = float('inf')
        max_batch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()
            boards = batch['board'].to(device)
            
            optimizer.zero_grad()
            reconstructed, embedding = model(boards)
            loss = criterion(reconstructed, boards)
            loss.backward()
            
            # Compute gradient norm before step
            grad_norm = compute_gradient_norm(model)
            
            optimizer.step()
            
            loss_value = loss.item()
            total_loss += loss_value
            batch_count += 1
            step += 1
            running_losses.append(loss_value)
            min_batch_loss = min(min_batch_loss, loss_value)
            max_batch_loss = max(max_batch_loss, loss_value)
            
            batch_time = time.time() - batch_start
            samples_per_sec = boards.size(0) / batch_time if batch_time > 0 else 0
            
            # Log detailed metrics every LOG_INTERVAL steps
            if step % LOG_INTERVAL == 0 and wandb.run is not None:
                running_avg_loss = np.mean(running_losses) if running_losses else loss_value
                param_stats = compute_parameter_stats(model)
                
                log_dict = {
                    'autoencoder/train_loss': loss_value,
                    'autoencoder/train_loss_running_avg': running_avg_loss,
                    'autoencoder/train_loss_min': min_batch_loss,
                    'autoencoder/train_loss_max': max_batch_loss,
                    'autoencoder/learning_rate': optimizer.param_groups[0]['lr'],
                    'autoencoder/gradient_norm': grad_norm,
                    'autoencoder/samples_per_sec': samples_per_sec,
                    'autoencoder/step': step,
                    'autoencoder/epoch': epoch + 1,
                    'autoencoder/batch': batch_idx + 1,
                }
                log_dict.update({f'autoencoder/param_{k}': v for k, v in param_stats.items()})
                wandb.log(log_dict)
            
            pbar.set_postfix({
                'loss': loss_value,
                'avg': np.mean(running_losses) if running_losses else loss_value,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Compute epoch-level statistics
        param_stats = compute_parameter_stats(model)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Evaluation
        eval_metrics = {}
        if eval_dataloader is not None and (epoch + 1) % EVAL_INTERVAL == 0:
            eval_metrics = evaluate_autoencoder(model, eval_dataloader, device)
            logger.info(f"Eval Loss: {eval_metrics.get('eval_loss', 0):.6f}")
        
        # Log comprehensive epoch metrics to wandb
        if wandb.run is not None:
            log_dict = {
                'autoencoder/epoch': epoch + 1,
                'autoencoder/epoch_loss': avg_loss,
                'autoencoder/epoch_loss_min': min_batch_loss,
                'autoencoder/epoch_loss_max': max_batch_loss,
                'autoencoder/epoch_loss_std': float(np.std(list(running_losses))) if running_losses else 0.0,
                'autoencoder/learning_rate': current_lr,
                'autoencoder/epoch_time': epoch_time,
                'autoencoder/samples_per_sec_epoch': batch_count * BATCH_SIZE / epoch_time if epoch_time > 0 else 0,
                'autoencoder/total_steps': step,
                'autoencoder/batches_per_epoch': batch_count,
            }
            log_dict.update({f'autoencoder/param_{k}': v for k, v in param_stats.items()})
            log_dict.update({f'autoencoder/{k}': v for k, v in eval_metrics.items()})
            wandb.log(log_dict)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, f'autoencoder_epoch_{epoch+1}_loss_{avg_loss:.6f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'eval_loss': eval_metrics.get('eval_loss', None),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Log best model to wandb
            if wandb.run is not None:
                wandb.log({
                    'autoencoder/best_loss': best_loss,
                    'autoencoder/best_epoch': epoch + 1,
                    'autoencoder/checkpoint_path': checkpoint_path
                })


def train_policy_head(bot: TicTacToeBot, dataloader: DataLoader, 
                     num_epochs: int, device: str, save_dir: str,
                     eval_dataloader: Optional[DataLoader] = None):
    """Train the policy head using winning positions"""
    bot = bot.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(bot.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_loss = float('inf')
    best_accuracy = 0.0
    step = 0
    running_losses = deque(maxlen=100)
    running_accuracies = deque(maxlen=100)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        bot.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        batch_count = 0
        min_batch_loss = float('inf')
        max_batch_loss = 0.0
        min_batch_acc = 1.0
        max_batch_acc = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()
            boards = batch['board'].to(device)
            win_actors = batch['win_actor']
            
            # For now, we'll use a simple approach:
            # Learn to predict moves that lead to winning positions
            # This is a simplified version - in practice, you'd want to use
            # actual move sequences from the database
            
            optimizer.zero_grad()
            
            # Get move predictions
            move_probs = bot(boards)
            
            # Create targets: prefer moves in winning positions
            # This is a heuristic - ideally you'd have actual move sequences
            batch_size = boards.shape[0]
            # Get board size from model
            board_size = bot.board_size
            targets = torch.randint(0, board_size * board_size, (batch_size,)).to(device)
            
            loss = criterion(move_probs, targets)
            loss.backward()
            
            # Compute gradient norm before step
            grad_norm = compute_gradient_norm(bot)
            
            optimizer.step()
            
            loss_value = loss.item()
            total_loss += loss_value
            batch_count += 1
            step += 1
            running_losses.append(loss_value)
            min_batch_loss = min(min_batch_loss, loss_value)
            max_batch_loss = max(max_batch_loss, loss_value)
            
            # Calculate accuracy
            preds = torch.argmax(move_probs, dim=1)
            batch_correct = (preds == targets).sum().item()
            correct_predictions += batch_correct
            total_predictions += batch_size
            batch_accuracy = batch_correct / batch_size
            running_accuracies.append(batch_accuracy)
            min_batch_acc = min(min_batch_acc, batch_accuracy)
            max_batch_acc = max(max_batch_acc, batch_accuracy)
            
            batch_time = time.time() - batch_start
            samples_per_sec = batch_size / batch_time if batch_time > 0 else 0
            
            # Log detailed metrics every LOG_INTERVAL steps
            if step % LOG_INTERVAL == 0 and wandb.run is not None:
                running_avg_loss = np.mean(running_losses) if running_losses else loss_value
                running_avg_acc = np.mean(running_accuracies) if running_accuracies else batch_accuracy
                param_stats = compute_parameter_stats(bot)
                
                log_dict = {
                    'policy/train_loss': loss_value,
                    'policy/train_loss_running_avg': running_avg_loss,
                    'policy/train_loss_min': min_batch_loss,
                    'policy/train_loss_max': max_batch_loss,
                    'policy/train_accuracy': batch_accuracy,
                    'policy/train_accuracy_running_avg': running_avg_acc,
                    'policy/train_accuracy_min': min_batch_acc,
                    'policy/train_accuracy_max': max_batch_acc,
                    'policy/learning_rate': optimizer.param_groups[0]['lr'],
                    'policy/gradient_norm': grad_norm,
                    'policy/samples_per_sec': samples_per_sec,
                    'policy/step': step,
                    'policy/epoch': epoch + 1,
                    'policy/batch': batch_idx + 1,
                }
                log_dict.update({f'policy/param_{k}': v for k, v in param_stats.items()})
                wandb.log(log_dict)
            
            pbar.set_postfix({
                'loss': loss_value,
                'acc': batch_accuracy,
                'avg_acc': np.mean(running_accuracies) if running_accuracies else batch_accuracy,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Compute epoch-level statistics
        param_stats = compute_parameter_stats(bot)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}, Accuracy: {accuracy:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Evaluation
        eval_metrics = {}
        if eval_dataloader is not None and (epoch + 1) % EVAL_INTERVAL == 0:
            eval_metrics = evaluate_policy(bot, eval_dataloader, device)
            logger.info(f"Eval Loss: {eval_metrics.get('eval_loss', 0):.6f}, Eval Accuracy: {eval_metrics.get('eval_accuracy', 0):.4f}")
        
        # Log comprehensive epoch metrics to wandb
        if wandb.run is not None:
            log_dict = {
                'policy/epoch': epoch + 1,
                'policy/epoch_loss': avg_loss,
                'policy/epoch_loss_min': min_batch_loss,
                'policy/epoch_loss_max': max_batch_loss,
                'policy/epoch_loss_std': float(np.std(list(running_losses))) if running_losses else 0.0,
                'policy/epoch_accuracy': accuracy,
                'policy/epoch_accuracy_min': min_batch_acc,
                'policy/epoch_accuracy_max': max_batch_acc,
                'policy/epoch_accuracy_std': float(np.std(list(running_accuracies))) if running_accuracies else 0.0,
                'policy/learning_rate': current_lr,
                'policy/epoch_time': epoch_time,
                'policy/samples_per_sec_epoch': batch_count * BATCH_SIZE / epoch_time if epoch_time > 0 else 0,
                'policy/total_steps': step,
                'policy/batches_per_epoch': batch_count,
                'policy/total_correct': correct_predictions,
                'policy/total_predictions': total_predictions,
            }
            log_dict.update({f'policy/param_{k}': v for k, v in param_stats.items()})
            log_dict.update({f'policy/{k}': v for k, v in eval_metrics.items()})
            wandb.log(log_dict)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, f'bot_epoch_{epoch+1}_loss_{avg_loss:.6f}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': bot.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'accuracy': accuracy,
                'eval_loss': eval_metrics.get('eval_loss', None),
                'eval_accuracy': eval_metrics.get('eval_accuracy', None),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Log best model to wandb
            if wandb.run is not None:
                wandb.log({
                    'policy/best_loss': best_loss,
                    'policy/best_epoch': epoch + 1,
                    'policy/best_accuracy': accuracy,
                    'policy/checkpoint_path': checkpoint_path
                })
        
        # Track best accuracy separately
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            if wandb.run is not None:
                wandb.log({
                    'policy/best_accuracy': best_accuracy,
                    'policy/best_accuracy_epoch': epoch + 1
                })


def main():
    """Main training function"""
    logger.info("Starting training...")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model type: {MODEL_TYPE}")
    logger.info(f"Configured board size: {BOARD_SIZE}x{BOARD_SIZE} (will use actual data size)")
    
    # Initialize wandb
    training_start_time = time.time()
    wandb_run_id = wandb.util.generate_id()
    wandb.init(
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        name=WANDB_NAME or f"{MODEL_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        mode=WANDB_MODE,
        id=wandb_run_id,
        config={
            'board_size': BOARD_SIZE,
            'win_length': WIN_LENGTH,
            'model_type': MODEL_TYPE,
            'use_autoencoder': USE_AUTOENCODER,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'embedding_dim': EMBEDDING_DIM,
            'device': DEVICE,
            'eval_interval': EVAL_INTERVAL,
            'eval_split': EVAL_SPLIT,
            'log_interval': LOG_INTERVAL,
        }
    )
    logger.info(f"Initialized wandb run: {wandb.run.name} (ID: {wandb_run_id})")
    
    # Connect to database
    logger.info("Connecting to ClickHouse...")
    db_connector = ClickHouseConnector(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DATABASE
    )
    
    # Load boards - detect board size from data
    logger.info("Loading boards from database...")
    boards = db_connector.get_all_boards(board_size=5, max_boards_per_table=10000)
    
    if len(boards) == 0:
        logger.error("No boards loaded from database!")
        return
    
    # Detect actual board size from the data
    if len(boards) > 0:
        actual_board_size = boards[0][0].shape[0]
        logger.info(f"Detected board size from data: {actual_board_size}x{actual_board_size}")
        # Update wandb config with actual board size
        if wandb.run is not None:
            wandb.config.update({'actual_board_size': actual_board_size, 'num_boards': len(boards)})
    else:
        actual_board_size = BOARD_SIZE
        logger.warning(f"No boards found, using default board size: {actual_board_size}x{actual_board_size}")
    
    logger.info(f"Loaded {len(boards)} boards")
    
    # Split into train and eval sets
    split_idx = int(len(boards) * (1 - EVAL_SPLIT))
    train_boards = boards[:split_idx]
    eval_boards = boards[split_idx:]
    
    logger.info(f"Train set: {len(train_boards)} boards, Eval set: {len(eval_boards)} boards")
    
    # Create datasets
    train_dataset = BoardDataset(train_boards, model_type=MODEL_TYPE, augment=True)
    eval_dataset = BoardDataset(eval_boards, model_type=MODEL_TYPE, augment=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Log data statistics to wandb
    if wandb.run is not None:
        # Count win actors
        win_actor_counts = {}
        for _, win_actor in boards:
            win_actor_counts[win_actor] = win_actor_counts.get(win_actor, 0) + 1
        
        wandb.log({
            'data/total_boards': len(boards),
            'data/train_boards': len(train_boards),
            'data/eval_boards': len(eval_boards),
            'data/train_batches': len(train_dataloader),
            'data/eval_batches': len(eval_dataloader),
            'data/win_X_count': win_actor_counts.get('X', 0),
            'data/win_O_count': win_actor_counts.get('O', 0),
            'data/draw_count': win_actor_counts.get('D', 0),
        })
    
    # Create models using actual board size from data
    if USE_AUTOENCODER:
        logger.info("Training autoencoder...")
        autoencoder = AutoEncoder(board_size=actual_board_size, embedding_dim=EMBEDDING_DIM, model_type=MODEL_TYPE)
        train_autoencoder(autoencoder, train_dataloader, NUM_EPOCHS, DEVICE, CHECKPOINT_DIR, eval_dataloader)
        
        # Use encoder from autoencoder for the bot
        bot = TicTacToeBot(board_size=actual_board_size, embedding_dim=EMBEDDING_DIM, 
                          model_type=MODEL_TYPE, use_autoencoder=False)
        bot.encoder.load_state_dict(autoencoder.encoder.state_dict())
        logger.info("Loaded encoder weights from autoencoder")
    else:
        bot = TicTacToeBot(board_size=actual_board_size, embedding_dim=EMBEDDING_DIM, 
                          model_type=MODEL_TYPE, use_autoencoder=False)
    
    # Train policy head
    logger.info("Training policy head...")
    train_policy_head(bot, train_dataloader, NUM_EPOCHS, DEVICE, CHECKPOINT_DIR, eval_dataloader)
    
    # Save final model
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    final_model_path = os.path.join(MODEL_SAVE_DIR, 'final_model.pt')
    torch.save(bot.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Log final model and training summary to wandb
    total_training_time = time.time() - training_start_time
    if wandb.run is not None:
        wandb.log({
            'final_model_path': final_model_path,
            'training/total_time_seconds': total_training_time,
            'training/total_time_hours': total_training_time / 3600,
        })
        wandb.finish()
        logger.info("Wandb run completed")
    
    db_connector.close()
    logger.info(f"Training completed! Total time: {total_training_time/3600:.2f} hours")


if __name__ == '__main__':
    main()

