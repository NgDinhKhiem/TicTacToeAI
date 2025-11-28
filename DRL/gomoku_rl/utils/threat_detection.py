"""
Threat detection utilities for Gomoku.
Detects double-three threats, double-open-three threats, open fours,
opponent forks, and other strategic factors.
"""
import torch
from typing import Tuple


def count_open_threes_for_move(
    board: torch.Tensor,
    r: int,
    c: int,
    player_piece: int,
    board_size: int,
) -> int:
    """
    Counts how many open three patterns are created by placing a piece at (r, c).
    An open three is a pattern _XXX_ (three consecutive pieces with empty on both sides).
    
    Args:
        board: Board tensor of shape (B, B)
        r: Row position
        c: Column position
        player_piece: 1 for black, -1 for white
        board_size: Size of the board
        
    Returns:
        Number of distinct open three patterns created
    """
    # Create temporary board with the move
    temp_board = board.clone()
    temp_board[r, c] = player_piece
    
    num_open_threes = 0
    directions = [
        (0, 1),   # horizontal
        (1, 0),   # vertical
        (1, 1),   # diagonal \
        (1, -1),  # diagonal /
    ]
    
    # Check each direction once (pattern _XXX_ is symmetric, so we only need to check one direction)
    for dr, dc in directions:
        # Look for pattern: empty, stone, stone, stone, empty
        # Pattern positions relative to (r, c): -1, 0, 1, 2, 3
        # Check if (r, c) can be part of such a pattern
        pattern_valid = True
        pattern_count = 0
        
        # Check the 5 positions in the positive direction
        for offset in range(-1, 4):
            nr = r + offset * dr
            nc = c + offset * dc
            
            if not (0 <= nr < board_size and 0 <= nc < board_size):
                pattern_valid = False
                break
            
            if offset == -1 or offset == 3:
                # Should be empty
                if temp_board[nr, nc] != 0:
                    pattern_valid = False
                    break
            else:
                # Should be player piece
                if temp_board[nr, nc] != player_piece:
                    pattern_valid = False
                    break
                pattern_count += 1
        
        if pattern_valid and pattern_count == 3:
            num_open_threes += 1
    
    return num_open_threes


def count_open_fours_for_move(
    board: torch.Tensor,
    r: int,
    c: int,
    player_piece: int,
    board_size: int,
) -> int:
    """
    Counts how many open four patterns are created by placing a piece at (r, c).
    An open four is a pattern _XXXX_ (four consecutive pieces with empty on both sides).
    
    Args:
        board: Board tensor of shape (B, B)
        r: Row position
        c: Column position
        player_piece: 1 for black, -1 for white
        board_size: Size of the board
        
    Returns:
        Number of distinct open four patterns created
    """
    temp_board = board.clone()
    temp_board[r, c] = player_piece
    
    num_open_fours = 0
    directions = [
        (0, 1),   # horizontal
        (1, 0),   # vertical
        (1, 1),   # diagonal \
        (1, -1),  # diagonal /
    ]
    
    for dr, dc in directions:
        # Look for pattern: empty, stone, stone, stone, stone, empty
        pattern_valid = True
        pattern_count = 0
        
        for offset in range(-1, 5):
            nr = r + offset * dr
            nc = c + offset * dc
            
            if not (0 <= nr < board_size and 0 <= nc < board_size):
                pattern_valid = False
                break
            
            if offset == -1 or offset == 4:
                # Should be empty
                if temp_board[nr, nc] != 0:
                    pattern_valid = False
                    break
            else:
                # Should be player piece
                if temp_board[nr, nc] != player_piece:
                    pattern_valid = False
                    break
                pattern_count += 1
        
        if pattern_valid and pattern_count == 4:
            num_open_fours += 1
    
    return num_open_fours


def detect_opponent_fork(
    board: torch.Tensor,
    r: int,
    c: int,
    opponent_piece: int,
    board_size: int,
) -> bool:
    """
    Detects if placing a piece at (r, c) would block an opponent's fork (double-three threat).
    A fork is when the opponent can create 2+ open threes with a single move.
    
    Args:
        board: Board tensor of shape (B, B)
        r: Row position where we would place our piece
        c: Column position where we would place our piece
        opponent_piece: -1 for black, 1 for white (opponent's piece value)
        board_size: Size of the board
        
    Returns:
        True if blocking at (r, c) prevents an opponent's fork
    """
    # Check if opponent currently has a fork that would be blocked by placing at (r, c)
    # We check if opponent can create 2+ open threes by placing at (r, c)
    # If yes, then blocking at (r, c) prevents that fork
    
    # Count how many open threes opponent would create by placing at (r, c)
    num_opponent_threes = count_open_threes_for_move(
        board, r, c, opponent_piece, board_size
    )
    
    # If opponent would create 2+ open threes at this position, blocking it prevents a fork
    return num_opponent_threes >= 2


def compute_center_control_boost(
    action_mask: torch.Tensor,
    board_size: int,
    device: torch.device,
    center_boost: float = 0.5,
) -> torch.Tensor:
    """
    Computes boost for moves that control the center of the board.
    
    Args:
        action_mask: Action mask of shape (E, B*B)
        board_size: Size of the board
        device: Device to run computations on
        center_boost: Boost value for center positions
        
    Returns:
        Boost tensor of shape (E, B*B)
    """
    num_envs = action_mask.shape[0]
    boost = torch.zeros(num_envs, board_size * board_size, device=device, dtype=torch.float32)
    
    center_r = board_size // 2
    center_c = board_size // 2
    
    for action_idx in range(board_size * board_size):
        r = action_idx // board_size
        c = action_idx % board_size
        
        # Calculate distance from center
        dist_r = abs(r - center_r)
        dist_c = abs(c - center_c)
        distance = (dist_r ** 2 + dist_c ** 2) ** 0.5
        
        # Maximum distance from center
        max_dist = ((board_size // 2) ** 2 + (board_size // 2) ** 2) ** 0.5
        
        # Boost inversely proportional to distance from center
        # Positions closer to center get higher boost
        if max_dist > 0:
            boost_value = center_boost * (1.0 - distance / max_dist)
        else:
            boost_value = center_boost
        
        # Only apply to valid actions
        boost[:, action_idx] = torch.where(
            action_mask[:, action_idx],
            boost_value,
            torch.tensor(0.0, device=device)
        )
    
    return boost


def compute_corner_edge_boost(
    action_mask: torch.Tensor,
    board_size: int,
    device: torch.device,
    corner_boost: float = 0.3,
    edge_boost: float = 0.2,
) -> torch.Tensor:
    """
    Computes boost for moves that control corners and edges (strategic positions).
    
    Args:
        action_mask: Action mask of shape (E, B*B)
        board_size: Size of the board
        device: Device to run computations on
        corner_boost: Boost value for corner positions
        edge_boost: Boost value for edge positions
        
    Returns:
        Boost tensor of shape (E, B*B)
    """
    num_envs = action_mask.shape[0]
    boost = torch.zeros(num_envs, board_size * board_size, device=device, dtype=torch.float32)
    
    corners = [
        (0, 0),
        (0, board_size - 1),
        (board_size - 1, 0),
        (board_size - 1, board_size - 1),
    ]
    
    for action_idx in range(board_size * board_size):
        r = action_idx // board_size
        c = action_idx % board_size
        
        boost_value = 0.0
        
        # Check if it's a corner
        if (r, c) in corners:
            boost_value = corner_boost
        # Check if it's on an edge (but not corner)
        elif r == 0 or r == board_size - 1 or c == 0 or c == board_size - 1:
            boost_value = edge_boost
        
        # Only apply to valid actions
        boost[:, action_idx] = torch.where(
            action_mask[:, action_idx],
            boost_value,
            torch.tensor(0.0, device=device)
        )
    
    return boost


def compute_comprehensive_strategic_boost(
    board: torch.Tensor,
    action_mask: torch.Tensor,
    player_piece: int,
    board_size: int,
    device: torch.device,
    # Threat parameters
    double_three_boost: float = 2.0,
    double_open_three_boost: float = 3.0,
    open_four_boost: float = 5.0,
    # Blocking parameters
    block_fork_boost: float = 4.0,
    block_open_four_boost: float = 6.0,
    # Positional parameters
    center_boost: float = 0.5,
    corner_boost: float = 0.3,
    edge_boost: float = 0.2,
) -> torch.Tensor:
    """
    Computes a comprehensive strategic boost tensor for actions.
    Includes: threats, blocking, center control, corners/edges.
    
    Args:
        board: Board tensor of shape (E, B, B) where E is num_envs, B is board_size
        action_mask: Action mask of shape (E, B*B) indicating valid actions
        player_piece: 1 for black, -1 for white
        board_size: Size of the board
        device: Device to run computations on
        double_three_boost: Boost for double-three threats
        double_open_three_boost: Boost for double-open-three threats
        open_four_boost: Boost for open four threats
        block_fork_boost: Boost for blocking opponent's fork
        block_open_four_boost: Boost for blocking opponent's open four
        center_boost: Boost for center positions
        corner_boost: Boost for corner positions
        edge_boost: Boost for edge positions
        
    Returns:
        Boost tensor of shape (E, B*B) to add to logits before softmax
    """
    num_envs = board.shape[0]
    boost = torch.zeros(num_envs, board_size * board_size, device=device, dtype=torch.float32)
    
    opponent_piece = -player_piece
    
    # For each environment and each valid action, evaluate strategic value
    for env_idx in range(num_envs):
        env_board = board[env_idx]  # (B, B)
        env_mask = action_mask[env_idx]  # (B*B,)
        
        # Get valid actions
        valid_actions = torch.where(env_mask)[0]
        
        for action in valid_actions:
            action_int = action.item()
            r = action_int // board_size
            c = action_int % board_size
            
            # Skip if position is not empty
            if env_board[r, c] != 0:
                continue
            
            action_boost = 0.0
            
            # 1. Check for open fours (highest priority - winning move)
            num_open_fours = count_open_fours_for_move(
                env_board, r, c, player_piece, board_size
            )
            if num_open_fours > 0:
                action_boost += open_four_boost * num_open_fours
            
            # 2. Check for double-open-three threats
            num_open_threes = count_open_threes_for_move(
                env_board, r, c, player_piece, board_size
            )
            if num_open_threes >= 2:
                action_boost += double_open_three_boost
            elif num_open_threes >= 1:
                action_boost += double_three_boost * 0.5
            
            # 3. Check for blocking opponent's open four (critical defense)
            opponent_open_fours = count_open_fours_for_move(
                env_board, r, c, opponent_piece, board_size
            )
            if opponent_open_fours > 0:
                action_boost += block_open_four_boost * opponent_open_fours
            
            # 4. Check for blocking opponent's fork (double-three threat)
            if detect_opponent_fork(env_board, r, c, opponent_piece, board_size):
                action_boost += block_fork_boost
            
            boost[env_idx, action_int] = action_boost
    
    # Add positional boosts (center, corners, edges)
    center_boost_tensor = compute_center_control_boost(
        action_mask, board_size, device, center_boost
    )
    corner_edge_boost_tensor = compute_corner_edge_boost(
        action_mask, board_size, device, corner_boost, edge_boost
    )
    
    boost = boost + center_boost_tensor + corner_edge_boost_tensor
    
    return boost


def compute_threat_boost(
    board: torch.Tensor,
    action_mask: torch.Tensor,
    player_piece: int,
    board_size: int,
    device: torch.device,
    double_three_boost: float = 2.0,
    double_open_three_boost: float = 3.0,
    open_four_boost: float = 5.0,
    block_fork_boost: float = 4.0,
    block_open_four_boost: float = 6.0,
    center_boost: float = 0.5,
    corner_boost: float = 0.3,
    edge_boost: float = 0.2,
) -> torch.Tensor:
    """
    Computes a comprehensive strategic boost tensor for actions.
    This is the main function to use - it includes all strategic factors:
    - Open threes and fours
    - Blocking opponent's forks and open fours
    - Center control
    - Corner and edge control
    
    Args:
        board: Board tensor of shape (E, B, B) where E is num_envs, B is board_size
        action_mask: Action mask of shape (E, B*B) indicating valid actions
        player_piece: 1 for black, -1 for white
        board_size: Size of the board
        device: Device to run computations on
        double_three_boost: Boost for double-three threats
        double_open_three_boost: Boost for double-open-three threats
        open_four_boost: Boost for open four threats
        block_fork_boost: Boost for blocking opponent's fork
        block_open_four_boost: Boost for blocking opponent's open four
        center_boost: Boost for center positions
        corner_boost: Boost for corner positions
        edge_boost: Boost for edge positions
        
    Returns:
        Boost tensor of shape (E, B*B) to add to logits before softmax
    """
    return compute_comprehensive_strategic_boost(
        board=board,
        action_mask=action_mask,
        player_piece=player_piece,
        board_size=board_size,
        device=device,
        double_three_boost=double_three_boost,
        double_open_three_boost=double_open_three_boost,
        open_four_boost=open_four_boost,
        block_fork_boost=block_fork_boost,
        block_open_four_boost=block_open_four_boost,
        center_boost=center_boost,
        corner_boost=corner_boost,
        edge_boost=edge_boost,
    )

