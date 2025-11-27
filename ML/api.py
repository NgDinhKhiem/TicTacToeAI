"""
API endpoint for ML-based Tic-Tac-Toe bot
"""

from flask import Flask, request, jsonify
import torch
import os
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ML.models import TicTacToeBot
    from ML.data_utils import board_to_tensor, create_valid_moves_mask
    from ML.config import *
except ImportError:
    # For running as script from ML directory
    from models import TicTacToeBot
    from data_utils import board_to_tensor, create_valid_moves_mask
    from config import *

app = Flask(__name__)

# Global model cache
_model_cache = {}
_imports_loaded = False


def _ensure_imports():
    """Ensure all imports are loaded"""
    global _imports_loaded
    if not _imports_loaded:
        _imports_loaded = True


def parse_board_matrix_ml(matrix: str, board_size: int):
    """
    Parse board matrix string to 2D list
    
    Args:
        matrix: String representation of board
        board_size: Size of the board
    
    Returns:
        2D list representing the board
    """
    # Remove whitespace
    matrix = matrix.replace(' ', '').replace('\n', '').replace('\r', '')
    
    # Handle different formats
    if ',' in matrix:
        values = matrix.split(',')
    elif '|' in matrix:
        values = matrix.split('|')
    else:
        # Single string, split by character
        values = list(matrix)
    
    board_2d = []
    for i in range(board_size):
        row = []
        for j in range(board_size):
            idx = i * board_size + j
            if idx < len(values):
                char = values[idx].upper()
                if char == 'X':
                    row.append('X')
                elif char == 'O':
                    row.append('O')
                else:
                    row.append('-')
            else:
                row.append('-')
        board_2d.append(row)
    
    return board_2d


def board_2d_to_numpy(board_2d, board_size: int):
    """Convert 2D list to numpy array"""
    import numpy as np
    board = np.zeros((board_size, board_size), dtype=np.int32)
    for i in range(board_size):
        for j in range(board_size):
            char = board_2d[i][j].upper()
            if char == 'X':
                board[i, j] = 1
            elif char == 'O':
                board[i, j] = 2
            else:
                board[i, j] = 0
    return board


def get_model(board_size: int = 10, checkpoint_path: str = None):
    """
    Get or load the model
    
    Args:
        board_size: Size of the board
        checkpoint_path: Optional path to checkpoint file
    
    Returns:
        Loaded model
    """
    cache_key = f"{board_size}_{checkpoint_path or 'default'}"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    # Create model
    model = TicTacToeBot(
        board_size=board_size,
        embedding_dim=EMBEDDING_DIM,
        model_type=MODEL_TYPE,
        use_autoencoder=USE_AUTOENCODER
    )
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logging.info(f"Loaded model from {checkpoint_path}")
        except Exception as e:
            logging.warning(f"Error loading checkpoint {checkpoint_path}: {e}. Using untrained model.")
    else:
        # Try to load default model
        default_model_path = os.path.join(os.path.dirname(__file__), MODEL_SAVE_DIR, 'final_model.pt')
        if os.path.exists(default_model_path):
            try:
                model.load_state_dict(torch.load(default_model_path, map_location='cpu'))
                logging.info(f"Loaded default model from {default_model_path}")
            except Exception as e:
                logging.warning(f"Error loading default model: {e}. Using untrained model.")
    
    model.eval()
    _model_cache[cache_key] = model
    return model


@app.route('/api/move', methods=['GET'])
def get_move():
    try:
        # Ensure imports are loaded before using them
        _ensure_imports()
        
        board_size = int(request.args.get('boardSize') or request.args.get('size') or '10')
        win_length = int(request.args.get('winLength') or request.args.get('win') or '5')
        next_move = (request.args.get('nextMove') or request.args.get('player') or 'X').upper()
        last_move_row_str = request.args.get('last_move_row')
        last_move_col_str = request.args.get('last_move_col')
        
        # Convert to integers if provided, otherwise None
        last_move_row = int(last_move_row_str) if last_move_row_str is not None else None
        last_move_col = int(last_move_col_str) if last_move_col_str is not None else None
        
        # Get board matrix from request
        matrix = request.args.get('matrix') or request.args.get('board')
        
        # Validate parameters
        if board_size < 5 or board_size > 20:
            return jsonify({'error': 'Board size must be between 5 and 20'}), 400
        if win_length < 5 or win_length > board_size:
            return jsonify({'error': f'Win length must be between 5 and {board_size}'}), 400
        if next_move not in ['X', 'O']:
            return jsonify({'error': 'Next move must be X or O'}), 400
        
        # Parse board matrix
        if matrix:
            try:
                board_2d = parse_board_matrix_ml(matrix, board_size)
            except Exception as e:
                return jsonify({'error': f'Invalid matrix format: {e}'}), 400
        else:
            board_2d = [['-' for _ in range(board_size)] for _ in range(board_size)]
        
        # Convert to numpy array
        import numpy as np
        board_np = board_2d_to_numpy(board_2d, board_size)
        
        # Get checkpoint path from request (optional)
        checkpoint_path = request.args.get('checkpoint')
        
        # Get model
        model = get_model(board_size, checkpoint_path)
        model.eval()
        
        # Convert board to tensor
        device = 'cpu'  # Use CPU for API
        board_tensor = board_to_tensor(board_np, MODEL_TYPE).to(device)
        
        # Create valid moves mask
        valid_moves_mask = create_valid_moves_mask(board_np).to(device)
        
        # Check if there are any valid moves
        if valid_moves_mask.sum().item() == 0:
            return jsonify({'error': 'No valid moves available'}), 400
        
        # Predict move
        with torch.no_grad():
            move_probs = model(board_tensor)
            
            # Apply valid moves mask
            move_probs = move_probs * valid_moves_mask.unsqueeze(0)
            # Set invalid moves to very negative value
            move_probs = move_probs + (1 - valid_moves_mask.unsqueeze(0)) * (-1e9)
            
            # Get best move
            best_move_idx = torch.argmax(move_probs, dim=1).item()
            best_row = best_move_idx // board_size
            best_col = best_move_idx % board_size
        
        # Validate the move
        if board_2d[best_row][best_col] != '-':
            # If invalid, find first valid move as fallback
            valid_indices = torch.nonzero(valid_moves_mask, as_tuple=False).squeeze(-1)
            if len(valid_indices) > 0:
                best_move_idx = valid_indices[0].item()
                best_row = best_move_idx // board_size
                best_col = best_move_idx % board_size
            else:
                return jsonify({'error': 'No valid moves available'}), 400
        
        return jsonify({
            'row': best_row,
            'col': best_col,
        })
    
    except Exception as e:
        import traceback
        logging.error(f"Error in get_move: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e) or 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    import sys
    port = int(os.getenv('API_PORT', 5050))  # Different port from DRL API
    host = os.getenv('API_HOST', '0.0.0.0')
    debug = os.getenv('API_DEBUG', 'False').lower() == 'true'
    
    logging.basicConfig(level=logging.INFO)
    print(f"Starting ML Flask API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

