"""
Model architectures for unsupervised learning Tic-Tac-Toe bot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoardEncoder(nn.Module):
    """Encoder to convert board state to embedding"""
    
    def __init__(self, board_size: int = 10, embedding_dim: int = 128, model_type: str = 'CNN'):
        super(BoardEncoder, self).__init__()
        self.board_size = board_size
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        
        if model_type == 'CNN':
            # CNN-based encoder
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 3 channels: X, O, empty
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc1 = nn.Linear(256 * 4 * 4, 512)
            self.fc2 = nn.Linear(512, embedding_dim)
            self.dropout = nn.Dropout(0.3)
        else:
            # MLP-based encoder
            input_size = board_size * board_size * 3  # One-hot encoded
            self.fc1 = nn.Linear(input_size, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, embedding_dim)
            self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Args:
            x: Board tensor of shape (batch, 3, board_size, board_size) for CNN
               or (batch, board_size * board_size * 3) for MLP
        """
        if self.model_type == 'CNN':
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
        else:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
        
        return x


class BoardDecoder(nn.Module):
    """Decoder to reconstruct board from embedding"""
    
    def __init__(self, board_size: int = 10, embedding_dim: int = 128, model_type: str = 'CNN'):
        super(BoardDecoder, self).__init__()
        self.board_size = board_size
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        
        if model_type == 'CNN':
            self.fc1 = nn.Linear(embedding_dim, 512)
            self.fc2 = nn.Linear(512, 256 * 4 * 4)
            self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
            self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
            self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
            self.upsample = nn.Upsample(size=(board_size, board_size), mode='bilinear', align_corners=False)
        else:
            output_size = board_size * board_size * 3
            self.fc1 = nn.Linear(embedding_dim, 256)
            self.fc2 = nn.Linear(256, 512)
            self.fc3 = nn.Linear(512, output_size)
            self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Args:
            x: Embedding tensor of shape (batch, embedding_dim)
        Returns:
            Reconstructed board tensor
        """
        if self.model_type == 'CNN':
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = x.view(x.size(0), 256, 4, 4)
            x = self.upsample(x)
            x = F.relu(self.deconv1(x))
            x = F.relu(self.deconv2(x))
            x = self.deconv3(x)
        else:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
        
        return x


class AutoEncoder(nn.Module):
    """Autoencoder for unsupervised learning of board representations"""
    
    def __init__(self, board_size: int = 10, embedding_dim: int = 128, model_type: str = 'CNN'):
        super(AutoEncoder, self).__init__()
        self.encoder = BoardEncoder(board_size, embedding_dim, model_type)
        self.decoder = BoardDecoder(board_size, embedding_dim, model_type)
        self.model_type = model_type
    
    def forward(self, x):
        embedding = self.encoder(x)
        reconstructed = self.decoder(embedding)
        return reconstructed, embedding


class PolicyHead(nn.Module):
    """Policy head to predict next move from board embedding"""
    
    def __init__(self, embedding_dim: int = 128, board_size: int = 10):
        super(PolicyHead, self).__init__()
        self.board_size = board_size
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, board_size * board_size)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, embedding):
        """
        Args:
            embedding: Board embedding of shape (batch, embedding_dim)
        Returns:
            Move probabilities of shape (batch, board_size * board_size)
        """
        x = F.relu(self.fc1(embedding))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class TicTacToeBot(nn.Module):
    """Complete model combining autoencoder and policy head"""
    
    def __init__(self, board_size: int = 10, embedding_dim: int = 128, 
                 model_type: str = 'CNN', use_autoencoder: bool = True):
        super(TicTacToeBot, self).__init__()
        self.board_size = board_size
        self.embedding_dim = embedding_dim
        self.use_autoencoder = use_autoencoder
        
        if use_autoencoder:
            self.encoder = BoardEncoder(board_size, embedding_dim, model_type)
        else:
            # Direct encoder without decoder
            self.encoder = BoardEncoder(board_size, embedding_dim, model_type)
        
        self.policy_head = PolicyHead(embedding_dim, board_size)
    
    def forward(self, board, return_embedding=False):
        """
        Args:
            board: Board tensor
            return_embedding: Whether to return embedding
        Returns:
            Move probabilities and optionally embedding
        """
        embedding = self.encoder(board)
        move_probs = self.policy_head(embedding)
        
        if return_embedding:
            return move_probs, embedding
        return move_probs
    
    def predict_move(self, board, valid_moves_mask=None):
        """
        Predict the best move given a board state
        
        Args:
            board: Board tensor of shape (1, 3, board_size, board_size) or (1, board_size * board_size * 3)
            valid_moves_mask: Optional mask of valid moves (1 for valid, 0 for invalid)
        
        Returns:
            Best move as (row, col) tuple
        """
        self.eval()
        with torch.no_grad():
            move_probs = self.forward(board)
            
            if valid_moves_mask is not None:
                move_probs = move_probs * valid_moves_mask
                # Set invalid moves to very negative value
                move_probs = move_probs + (1 - valid_moves_mask) * (-1e9)
            
            best_move_idx = torch.argmax(move_probs, dim=1).item()
            row = best_move_idx // self.board_size
            col = best_move_idx % self.board_size
            
            return row, col

