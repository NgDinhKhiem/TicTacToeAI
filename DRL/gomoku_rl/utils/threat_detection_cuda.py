"""
CUDA-optimized threat detection for Gomoku.
Fully vectorized implementation that runs entirely on GPU with no CPU synchronization.
"""
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def compute_positional_boost_cuda(
    board_size: int,
    action_mask: torch.Tensor,
    center_boost: float = 0.5,
    corner_boost: float = 0.3,
    edge_boost: float = 0.2,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute positional boosts (center, corners, edges) fully vectorized on GPU.
    
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
    
    # Create coordinate grids
    r = torch.arange(board_size, device=device).unsqueeze(1).expand(board_size, board_size)
    c = torch.arange(board_size, device=device).unsqueeze(0).expand(board_size, board_size)
    
    # Center boost - inversely proportional to distance from center
    center_dist = ((r - board_size // 2) ** 2 + (c - board_size // 2) ** 2).sqrt()
    max_dist = ((board_size // 2) ** 2 + (board_size // 2) ** 2) ** 0.5
    center_boost_map = center_boost * (1 - center_dist / max_dist)
    
    # Corners and edges
    corner_mask = torch.zeros(board_size, board_size, device=device, dtype=torch.float32)
    corner_mask[0, 0] = corner_mask[0, -1] = corner_mask[-1, 0] = corner_mask[-1, -1] = corner_boost
    
    edge_mask = torch.zeros(board_size, board_size, device=device, dtype=torch.float32)
    edge_mask[0, :] = edge_mask[-1, :] = edge_mask[:, 0] = edge_mask[:, -1] = edge_boost
    edge_mask[0, 0] = edge_mask[0, -1] = edge_mask[-1, 0] = edge_mask[-1, -1] = 0.0
    
    positional_map = center_boost_map + corner_mask + edge_mask
    boost = positional_map.flatten().unsqueeze(0).expand(action_mask.shape[0], -1) * action_mask.float()
    
    return boost


def compute_threat_boost_cuda(
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
    CUDA-optimized threat detection that runs entirely on GPU.
    Uses simplified heuristics that can be vectorized efficiently.
    
    Args:
        board: Board tensor of shape (E, B, B) where E is num_envs, B is board_size
        action_mask: Action mask of shape (E, B*B) indicating valid actions
        player_piece: 1 for black, -1 for white
        board_size: Size of the board
        device: Device to run computations on
        
    Returns:
        Boost tensor of shape (E, B*B) to add to logits before softmax
    """
    num_envs = board.shape[0]
    opponent_piece = -player_piece
    
    # Ensure everything is on the correct device
    board = board.to(device)
    action_mask = action_mask.to(device)
    
    # Initialize boost tensor
    boost = torch.zeros(num_envs, board_size * board_size, device=device, dtype=torch.float32)
    
    # Add positional boosts (center, corners, edges) - fully vectorized
    pos_boost = compute_positional_boost_cuda(
        board_size, action_mask, center_boost, corner_boost, edge_boost, device
    )
    boost = boost + pos_boost
    
    # For pattern detection, we'll use a simplified approach that's GPU-friendly
    # Instead of checking every action individually, we'll use convolution-based detection
    
    # Create player and opponent masks
    player_mask = (board == player_piece).float()  # (E, B, B)
    opponent_mask = (board == opponent_piece).float()  # (E, B, B)
    empty_mask = (board == 0).float()  # (E, B, B)
    
    # Define direction kernels for pattern detection
    # We'll detect sequences of pieces in 4 directions
    
    # Horizontal patterns (3 in a row, 4 in a row)
    kernel_h3 = torch.tensor([[[1, 1, 1]]], device=device, dtype=torch.float32)  # 1x3
    kernel_h4 = torch.tensor([[[1, 1, 1, 1]]], device=device, dtype=torch.float32)  # 1x4
    
    # Vertical patterns
    kernel_v3 = torch.tensor([[[1], [1], [1]]], device=device, dtype=torch.float32)  # 3x1
    kernel_v4 = torch.tensor([[[1], [1], [1], [1]]], device=device, dtype=torch.float32)  # 4x1
    
    # For each environment, detect patterns using convolution
    # This is a simplified version that detects concentrations of pieces
    
    # Detect 3-in-a-row patterns (horizontal and vertical)
    # Use 'same' padding to keep output size = input size
    player_h3 = F.conv2d(player_mask.unsqueeze(1), kernel_h3.unsqueeze(0), padding=(0, 1))
    player_v3 = F.conv2d(player_mask.unsqueeze(1), kernel_v3.unsqueeze(0), padding=(1, 0))
    
    # Detect 4-in-a-row patterns
    # For kernel size 4, we need padding to make output match input size
    # padding = (kernel_size - 1) // 2, but for even kernels we need asymmetric padding
    player_h4 = F.conv2d(player_mask.unsqueeze(1), kernel_h4.unsqueeze(0), padding=(0, 1))
    player_v4 = F.conv2d(player_mask.unsqueeze(1), kernel_v4.unsqueeze(0), padding=(1, 0))
    
    # Same for opponent
    opponent_h3 = F.conv2d(opponent_mask.unsqueeze(1), kernel_h3.unsqueeze(0), padding=(0, 1))
    opponent_v3 = F.conv2d(opponent_mask.unsqueeze(1), kernel_v3.unsqueeze(0), padding=(1, 0))
    opponent_h4 = F.conv2d(opponent_mask.unsqueeze(1), kernel_h4.unsqueeze(0), padding=(0, 1))
    opponent_v4 = F.conv2d(opponent_mask.unsqueeze(1), kernel_v4.unsqueeze(0), padding=(1, 0))
    
    # Crop to ensure all have the same size (B, B)
    # The outputs might be slightly different due to padding, so crop to minimum size
    min_h = min(player_h3.shape[2], player_v3.shape[2], player_h4.shape[2], player_v4.shape[2])
    min_w = min(player_h3.shape[3], player_v3.shape[3], player_h4.shape[3], player_v4.shape[3])
    
    player_h3 = player_h3[:, :, :min_h, :min_w]
    player_v3 = player_v3[:, :, :min_h, :min_w]
    player_h4 = player_h4[:, :, :min_h, :min_w]
    player_v4 = player_v4[:, :, :min_h, :min_w]
    
    opponent_h3 = opponent_h3[:, :, :min_h, :min_w]
    opponent_v3 = opponent_v3[:, :, :min_h, :min_w]
    opponent_h4 = opponent_h4[:, :, :min_h, :min_w]
    opponent_v4 = opponent_v4[:, :, :min_h, :min_w]
    
    # Also crop the masks to match
    empty_mask = empty_mask[:, :min_h, :min_w]
    
    # Combine patterns (max across directions)
    player_3_pattern = torch.max(player_h3, player_v3).squeeze(1)  # (E, H, W)
    player_4_pattern = torch.max(player_h4, player_v4).squeeze(1)  # (E, H, W)
    opponent_3_pattern = torch.max(opponent_h3, opponent_v3).squeeze(1)  # (E, H, W)
    opponent_4_pattern = torch.max(opponent_h4, opponent_v4).squeeze(1)  # (E, H, W)
    
    # Create a dilation kernel to find positions adjacent to patterns
    dilation_kernel = torch.ones(1, 1, 3, 3, device=device, dtype=torch.float32)
    
    # Find positions near player's 3-in-a-row (potential to create 4)
    near_player_3 = F.conv2d(
        (player_3_pattern >= 2.5).float().unsqueeze(1),  # At least 3 pieces
        dilation_kernel,
        padding=1
    ).squeeze(1)
    
    # Find positions near player's 4-in-a-row (winning moves)
    near_player_4 = F.conv2d(
        (player_4_pattern >= 3.5).float().unsqueeze(1),  # At least 4 pieces
        dilation_kernel,
        padding=1
    ).squeeze(1)
    
    # Find positions near opponent's 3-in-a-row (need to block)
    near_opponent_3 = F.conv2d(
        (opponent_3_pattern >= 2.5).float().unsqueeze(1),
        dilation_kernel,
        padding=1
    ).squeeze(1)
    
    # Find positions near opponent's 4-in-a-row (critical block)
    near_opponent_4 = F.conv2d(
        (opponent_4_pattern >= 3.5).float().unsqueeze(1),
        dilation_kernel,
        padding=1
    ).squeeze(1)
    
    # Apply boosts based on pattern proximity (only on empty positions)
    pattern_boost = torch.zeros(num_envs, min_h, min_w, device=device, dtype=torch.float32)
    
    # Winning moves (near own 4-in-a-row)
    pattern_boost += (near_player_4 > 0).float() * empty_mask * open_four_boost
    
    # Critical blocks (near opponent's 4-in-a-row)
    pattern_boost += (near_opponent_4 > 0).float() * empty_mask * block_open_four_boost
    
    # Offensive moves (near own 3-in-a-row)
    pattern_boost += (near_player_3 > 0).float() * empty_mask * double_three_boost
    
    # Defensive moves (near opponent's 3-in-a-row)
    pattern_boost += (near_opponent_3 > 0).float() * empty_mask * block_fork_boost
    
    # Flatten and add to boost
    pattern_boost_flat = pattern_boost.view(num_envs, -1)
    
    # Pad pattern_boost_flat if it's smaller than board_size * board_size
    expected_size = board_size * board_size
    if pattern_boost_flat.shape[1] < expected_size:
        padding_size = expected_size - pattern_boost_flat.shape[1]
        pattern_boost_flat = F.pad(pattern_boost_flat, (0, padding_size), value=0.0)
    elif pattern_boost_flat.shape[1] > expected_size:
        pattern_boost_flat = pattern_boost_flat[:, :expected_size]
    
    boost = boost + pattern_boost_flat
    
    # Apply action mask (zero out invalid actions)
    boost = boost * action_mask.float()
    
    if debug and step_count % 100 == 0:
        logger.debug(f"[Step {step_count}] CUDA threat detection - "
                    f"Max boost: {boost.max().item():.2f}, "
                    f"Mean boost (valid): {boost[action_mask].mean().item():.4f}, "
                    f"Non-zero boosts: {(boost > 0).sum().item()}/{action_mask.sum().item()}")
    
    return boost

