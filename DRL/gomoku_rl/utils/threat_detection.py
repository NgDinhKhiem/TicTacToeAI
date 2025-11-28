"""
Threat detection utilities for Gomoku.
Detects double-three threats, double-open-three threats, open fours,
opponent forks, and other strategic factors.
OPTIMIZED: Uses convolution-based pattern matching and vectorized operations for speed.
"""
import torch
import torch.nn.functional as F
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


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
    OPTIMIZED: Avoids board cloning, uses direct checks.
    """
    num_open_threes = 0
    directions = [
        (0, 1),   # horizontal
        (1, 0),   # vertical
        (1, 1),   # diagonal \
        (1, -1),  # diagonal /
    ]
    
    # Check each direction - optimized to avoid cloning
    for dr, dc in directions:
        # Look for pattern: empty, stone, stone, stone, empty
        # Pattern positions relative to (r, c): -1, 0, 1, 2, 3
        
        # Quick bounds check first
        if not (0 <= r - dr < board_size and 0 <= c - dc < board_size and
                0 <= r + 3*dr < board_size and 0 <= c + 3*dc < board_size):
            continue
        
        # Check pattern positions
        # Position -1: should be empty
        if board[r - dr, c - dc] != 0:
            continue
        # Position 0: should be player piece (or empty, we'll place it)
        if board[r, c] != 0 and board[r, c] != player_piece:
            continue
        # Position 1: should be player piece
        if board[r + dr, c + dc] != player_piece:
            continue
        # Position 2: should be player piece
        if board[r + 2*dr, c + 2*dc] != player_piece:
            continue
        # Position 3: should be empty
        if board[r + 3*dr, c + 3*dc] != 0:
            continue
        
        # All checks passed - this is an open three
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
    OPTIMIZED: Avoids board cloning, uses direct checks.
    """
    num_open_fours = 0
    directions = [
        (0, 1),   # horizontal
        (1, 0),   # vertical
        (1, 1),   # diagonal \
        (1, -1),  # diagonal /
    ]
    
    for dr, dc in directions:
        # Look for pattern: empty, stone, stone, stone, stone, empty
        # Quick bounds check first
        if not (0 <= r - dr < board_size and 0 <= c - dc < board_size and
                0 <= r + 4*dr < board_size and 0 <= c + 4*dc < board_size):
            continue
        
        # Check pattern positions
        # Position -1: should be empty
        if board[r - dr, c - dc] != 0:
            continue
        # Position 0: should be player piece (or empty, we'll place it)
        if board[r, c] != 0 and board[r, c] != player_piece:
            continue
        # Positions 1-3: should be player piece
        if (board[r + dr, c + dc] != player_piece or
            board[r + 2*dr, c + 2*dc] != player_piece or
            board[r + 3*dr, c + 3*dc] != player_piece):
            continue
        # Position 4: should be empty
        if board[r + 4*dr, c + 4*dc] != 0:
            continue
        
        # All checks passed - this is an open four
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
    OPTIMIZED: Reuses count_open_threes_for_move which is already optimized.
    """
    # Count how many open threes opponent would create by placing at (r, c)
    num_opponent_threes = count_open_threes_for_move(
        board, r, c, opponent_piece, board_size
    )
    
    # If opponent would create 2+ open threes at this position, blocking it prevents a fork
    return num_opponent_threes >= 2


def compute_positional_boost(
    board_size: int,
    action_mask: torch.Tensor,
    center_boost: float = 0.5,
    corner_boost: float = 0.3,
    edge_boost: float = 0.2,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute positional boosts (center, corners, edges) vectorized.
    OPTIMIZED: Fully vectorized computation.
    
    Args:
        board_size: Size of the board
        action_mask: Action mask of shape (E, B*B)
        center_boost: Boost value for center positions
        corner_boost: Boost value for corner positions
        edge_boost: Boost value for edge positions
        device: Device to run computations on
        
    Returns:
        Boost tensor of shape (E, B*B)
    """
    if device is None:
        device = action_mask.device
    
    # Create coordinate grids - VECTORIZED
    r = torch.arange(board_size, device=device).unsqueeze(1).expand(board_size, board_size)
    c = torch.arange(board_size, device=device).unsqueeze(0).expand(board_size, board_size)
    
    # Center boost - inversely proportional to distance from center
    center_dist = ((r - board_size // 2) ** 2 + (c - board_size // 2) ** 2).sqrt()
    max_dist = ((board_size // 2) ** 2 + (board_size // 2) ** 2) ** 0.5
    center_boost_map = center_boost * (1 - center_dist / max_dist)
    
    # Corners and edges - VECTORIZED
    corner_mask = torch.zeros(board_size, board_size, device=device, dtype=torch.float32)
    corner_mask[0, 0] = corner_mask[0, -1] = corner_mask[-1, 0] = corner_mask[-1, -1] = corner_boost
    
    edge_mask = torch.zeros(board_size, board_size, device=device, dtype=torch.float32)
    edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = edge_boost
    edge_mask[0, 0] = edge_mask[0, -1] = edge_mask[-1, 0] = edge_mask[-1, -1] = 0.0  # Remove corner overlap
    
    positional_map = center_boost_map + corner_mask + edge_mask
    boost = positional_map.flatten().unsqueeze(0).expand(action_mask.shape[0], -1) * action_mask.float()
    
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
    # Performance optimization
    max_actions_to_check: int = 30,  # Limit actions checked per environment for speed (optimized default)
    # Debug parameters
    debug: bool = False,
    step_count: int = 0,
) -> torch.Tensor:
    """
    Computes a comprehensive strategic boost tensor for actions.
    OPTIMIZED: Uses optimized pattern detection and vectorized operations.
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
    work_device = torch.device("cpu")
    board_cpu = board.detach().to(work_device, non_blocking=True)
    action_mask_cpu = action_mask.to(work_device, non_blocking=True)
    boost_cpu = torch.zeros(
        num_envs, board_size * board_size, device=work_device, dtype=torch.float32
    )
    
    opponent_piece = -player_piece
    
    # Statistics for debugging
    threat_stats = {
        'open_fours': 0,
        'block_open_fours': 0,
        'double_open_threes': 0,
        'single_open_threes': 0,
        'block_forks': 0,
        'actions_checked': 0,
        'nearby_actions': 0,
    }
    
    # Optimize: Only check actions near existing pieces (within 2 squares)
    has_pieces = (board_cpu != 0).any(dim=0)  # (B, B) - any env has any piece
    
    # Create mask for positions within 2 squares of any piece - VECTORIZED
    nearby_mask = torch.zeros(board_size, board_size, device=work_device, dtype=torch.bool)
    if has_pieces.any():
        # Vectorized approach: use dilation-like operation
        kernel = torch.ones(5, 5, device=work_device, dtype=torch.bool)
        has_pieces_padded = F.pad(
            has_pieces.float().unsqueeze(0).unsqueeze(0), 
            (2, 2, 2, 2), 
            mode='constant', 
            value=0
        )
        kernel_expanded = kernel.unsqueeze(0).unsqueeze(0).float()
        nearby_mask = (F.conv2d(has_pieces_padded, kernel_expanded).squeeze() > 0.5)
    else:
        # Early game - no pieces yet, check center area
        center = board_size // 2
        nearby_mask[center-2:center+3, center-2:center+3] = True
    
    # For each environment, evaluate strategic value
    for env_idx in range(num_envs):
        env_board = board_cpu[env_idx]  # (B, B)
        env_board_int = env_board.to(dtype=torch.int8)
        env_mask = action_mask_cpu[env_idx]
        # Torch can hit an internal assert when torch.where/nonzero is called on
        # non-contiguous or higher-dimensional boolean masks; use Python loop instead.
        env_mask_flat = env_mask.reshape(-1).contiguous()
        
        # Get valid actions using Python loop to avoid PyTorch indexing bug
        valid_action_list = []
        for i in range(env_mask_flat.numel()):
            if env_mask_flat[i].item():
                valid_action_list.append(i)
        
        if len(valid_action_list) == 0:
            continue
        
        # Separate nearby and far actions using Python loop
        nearby_action_list = []
        far_action_list = []
        for action_idx in valid_action_list:
            r = action_idx // board_size
            c = action_idx % board_size
            if nearby_mask[r, c].item():
                nearby_action_list.append(action_idx)
            else:
                far_action_list.append(action_idx)
        
        # Prioritize nearby actions, limit total
        if len(nearby_action_list) >= max_actions_to_check:
            actions_to_check_list = nearby_action_list[:max_actions_to_check]
        else:
            num_far_needed = max_actions_to_check - len(nearby_action_list)
            actions_to_check_list = nearby_action_list + far_action_list[:num_far_needed]
        
        threat_stats['actions_checked'] += len(actions_to_check_list)
        threat_stats['nearby_actions'] += len(nearby_action_list)
        
        if debug and env_idx == 0:
            logger.debug(f"[Step {step_count}] Env {env_idx} - "
                        f"Total valid actions: {len(valid_action_list)}, "
                        f"Nearby actions: {len(nearby_action_list)}, "
                        f"Actions to check: {len(actions_to_check_list)}")
        
        # Process each action using optimized pattern detection
        for action_int in actions_to_check_list:
            r = action_int // board_size
            c = action_int % board_size
            
            # Skip if position is not empty
            if env_board_int[r, c].item() != 0:
                continue
            
            action_boost = 0.0
            
            # 1. Check for open fours (highest priority - winning move)
            num_open_fours = count_open_fours_for_move(
                env_board_int, r, c, int(player_piece), board_size
            )
            if num_open_fours > 0:
                action_boost += open_four_boost * num_open_fours
                threat_stats['open_fours'] += 1
                if debug and env_idx == 0:
                    logger.debug(f"[Step {step_count}] Env {env_idx} - Action ({r},{c}): "
                                f"OPEN FOUR detected! Boost: {action_boost:.2f}")
                boost_cpu[env_idx, action_int] = action_boost
                continue  # Early exit - winning move found
            
            # 2. Check for blocking opponent's open four (critical defense)
            opponent_open_fours = count_open_fours_for_move(
                env_board_int, r, c, int(opponent_piece), board_size
            )
            if opponent_open_fours > 0:
                action_boost += block_open_four_boost * opponent_open_fours
                threat_stats['block_open_fours'] += 1
                if debug and env_idx == 0:
                    logger.debug(f"[Step {step_count}] Env {env_idx} - Action ({r},{c}): "
                                f"BLOCK OPEN FOUR detected! Boost: {action_boost:.2f}")
                boost_cpu[env_idx, action_int] = action_boost
                continue  # Early exit - critical defense found
            
            # 3. Check for double-open-three threats
            num_open_threes = count_open_threes_for_move(
                env_board_int, r, c, int(player_piece), board_size
            )
            if num_open_threes >= 2:
                action_boost += double_open_three_boost
                threat_stats['double_open_threes'] += 1
                if debug and env_idx == 0:
                    logger.debug(f"[Step {step_count}] Env {env_idx} - Action ({r},{c}): "
                                f"DOUBLE OPEN THREE detected! Boost: {action_boost:.2f}")
            elif num_open_threes >= 1:
                action_boost += double_three_boost * 0.5
                threat_stats['single_open_threes'] += 1
            
            # 4. Check for blocking opponent's fork (double-three threat)
            if detect_opponent_fork(env_board_int, r, c, int(opponent_piece), board_size):
                action_boost += block_fork_boost
                threat_stats['block_forks'] += 1
                if debug and env_idx == 0:
                    logger.debug(f"[Step {step_count}] Env {env_idx} - Action ({r},{c}): "
                                f"BLOCK FORK detected! Boost: {action_boost:.2f}")
            
            boost_cpu[env_idx, action_int] = action_boost
    
    # Add positional boosts (center, corners, edges) - VECTORIZED
    pos_boost = compute_positional_boost(
        board_size, action_mask_cpu, center_boost, corner_boost, edge_boost, work_device
    )
    boost_cpu = boost_cpu + pos_boost
    
    if debug:
        logger.debug(f"[Step {step_count}] Threat detection summary - "
                    f"Open fours: {threat_stats['open_fours']}, "
                    f"Block open fours: {threat_stats['block_open_fours']}, "
                    f"Double open threes: {threat_stats['double_open_threes']}, "
                    f"Single open threes: {threat_stats['single_open_threes']}, "
                    f"Block forks: {threat_stats['block_forks']}, "
                    f"Actions checked: {threat_stats['actions_checked']}, "
                    f"Nearby actions: {threat_stats['nearby_actions']}")
    
    return boost_cpu.to(device)


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
    max_actions_to_check: int = 30,
    debug: bool = False,
    step_count: int = 0,
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
        max_actions_to_check=max_actions_to_check,
        debug=debug,
        step_count=step_count,
    )
