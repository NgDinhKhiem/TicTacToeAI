"""
Data utilities for preprocessing board states
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def board_to_tensor(board: np.ndarray, model_type: str = 'CNN') -> torch.Tensor:
    """
    Convert numpy board array to PyTorch tensor
    
    Args:
        board: numpy array of shape (board_size, board_size) with values 0, 1, 2
        model_type: 'CNN' or 'MLP'
    
    Returns:
        Tensor ready for model input
    """
    # Ensure board is contiguous to avoid negative stride issues
    board = np.ascontiguousarray(board)
    board_size = board.shape[0]
    
    if model_type == 'CNN':
        # Create 3-channel representation: [X_channel, O_channel, empty_channel]
        x_channel = (board == 1).astype(np.float32)
        o_channel = (board == 2).astype(np.float32)
        empty_channel = (board == 0).astype(np.float32)
        
        tensor = np.stack([x_channel, o_channel, empty_channel], axis=0)
        return torch.FloatTensor(tensor).unsqueeze(0)  # Add batch dimension
    else:
        # MLP: flatten and one-hot encode
        one_hot = np.zeros((board_size * board_size, 3), dtype=np.float32)
        for i in range(board_size):
            for j in range(board_size):
                idx = i * board_size + j
                one_hot[idx, int(board[i, j])] = 1.0
        
        tensor = one_hot.flatten()
        return torch.FloatTensor(tensor).unsqueeze(0)  # Add batch dimension


def boards_to_batch(boards: List[np.ndarray], model_type: str = 'CNN') -> torch.Tensor:
    """
    Convert list of boards to a batch tensor
    
    Args:
        boards: List of numpy arrays
        model_type: 'CNN' or 'MLP'
    
    Returns:
        Batched tensor
    """
    tensors = [board_to_tensor(board, model_type) for board in boards]
    return torch.cat(tensors, dim=0)


def parse_board_matrix(matrix_str: str, board_size: int) -> np.ndarray:
    """
    Parse board matrix string to numpy array
    
    Args:
        matrix_str: String representation of board (e.g., "X-O-X-O-...")
        board_size: Size of the board
    
    Returns:
        numpy array of shape (board_size, board_size)
    """
    # Remove whitespace and split
    matrix_str = matrix_str.replace(' ', '').replace('\n', '')
    
    # Handle different formats
    if ',' in matrix_str:
        # Comma-separated
        values = matrix_str.split(',')
    else:
        # Single string
        values = list(matrix_str)
    
    board = np.zeros((board_size, board_size), dtype=np.int32)
    
    for i in range(board_size):
        for j in range(board_size):
            idx = i * board_size + j
            if idx < len(values):
                char = values[idx].upper()
                if char == 'X':
                    board[i, j] = 1
                elif char == 'O':
                    board[i, j] = 2
                else:
                    board[i, j] = 0
    
    return board


def create_valid_moves_mask(board: np.ndarray) -> torch.Tensor:
    """
    Create a mask of valid moves (empty cells)
    
    Args:
        board: numpy array of shape (board_size, board_size)
    
    Returns:
        Tensor of shape (board_size * board_size,) with 1 for valid moves, 0 otherwise
    """
    board_size = board.shape[0]
    mask = (board == 0).flatten().astype(np.float32)
    return torch.FloatTensor(mask)


def augment_board(board: np.ndarray) -> List[np.ndarray]:
    """
    Generate augmented versions of a board (rotations and reflections)
    
    Args:
        board: numpy array of shape (board_size, board_size)
    
    Returns:
        List of augmented boards (all copied to avoid negative strides)
    """
    # Ensure input is a copy to avoid issues
    board = board.copy()
    augmented = [board]
    
    # Rotations - use copy() to ensure contiguous arrays
    for k in range(1, 4):
        rotated = np.rot90(board, k).copy()
        augmented.append(rotated)
    
    # Reflections - use copy() to ensure contiguous arrays
    augmented.append(np.fliplr(board).copy())
    augmented.append(np.flipud(board).copy())
    
    return augmented


def get_next_boards_from_sequence(boards: List[np.ndarray], win_actor: str) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Extract (current_board, next_board, move) tuples from a sequence of boards
    
    This is used for self-supervised learning where we learn to predict
    the next move that leads to a winning position.
    
    Args:
        boards: List of board states in sequence
        win_actor: 'X', 'O', or 'D' (draw)
    
    Returns:
        List of (current_board, next_board, move_index) tuples
    """
    if len(boards) < 2:
        return []
    
    results = []
    board_size = boards[0].shape[0]
    
    for i in range(len(boards) - 1):
        current = boards[i]
        next_board = boards[i + 1]
        
        # Find the move that was made
        diff = next_board - current
        move_positions = np.where(diff != 0)
        
        if len(move_positions[0]) > 0:
            row, col = move_positions[0][0], move_positions[1][0]
            move_idx = row * board_size + col
            results.append((current, next_board, move_idx))
    
    return results

