from flask import Flask, request, jsonify
import torch
from omegaconf import DictConfig, OmegaConf
import os
import json
import logging
import sys
import importlib.util

# Calculate CONFIG_PATH directly to avoid importing gomoku_rl.__init__ which triggers problematic imports
CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "cfg")
)

# Lazy import to avoid problematic import chain
# We'll import these modules only when needed, after Flask app is set up
# This helps avoid the tensordict.memmap import error that occurs during package initialization

app = Flask(__name__)

# Global model cache
_model_cache = {}
_config_cache = None

# Lazy import flag
_imports_loaded = False
_imports_failed = False
_import_error_message = None


def _load_imports():
    """Lazy load imports to avoid problematic import chain."""
    global _imports_loaded, _imports_failed, _import_error_message
    
    # If already loaded, return
    if _imports_loaded:
        return
    
    # If import already failed, raise the cached error
    if _imports_failed:
        raise ImportError(_import_error_message)
    
    # Suppress the warning about torchrl C++ binaries
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchrl")
    
    # Try to import modules directly to avoid gomoku_rl.__init__.py which imports GomokuEnv
    # This bypasses the problematic import chain
    try:
        gomoku_rl_path = os.path.join(os.path.dirname(__file__), "gomoku_rl")
        import types
        
        # Set up the gomoku_rl package structure
        if "gomoku_rl" not in sys.modules:
            sys.modules["gomoku_rl"] = types.ModuleType("gomoku_rl")
        if "gomoku_rl.utils" not in sys.modules:
            sys.modules["gomoku_rl.utils"] = types.ModuleType("gomoku_rl.utils")
        if "gomoku_rl.policy" not in sys.modules:
            sys.modules["gomoku_rl.policy"] = types.ModuleType("gomoku_rl.policy")
        
        # Import core module directly (doesn't import torchrl)
        core_spec = importlib.util.spec_from_file_location(
            "gomoku_rl.core",
            os.path.join(gomoku_rl_path, "core.py")
        )
        core_module = importlib.util.module_from_spec(core_spec)
        sys.modules["gomoku_rl.core"] = core_module
        core_spec.loader.exec_module(core_module)
        Gomoku = core_module.Gomoku
        compute_done = core_module.compute_done
        
        # Import utils modules that policy depends on
        # First, import utils.module
        utils_module_spec = importlib.util.spec_from_file_location(
            "gomoku_rl.utils.module",
            os.path.join(gomoku_rl_path, "utils", "module.py")
        )
        utils_module_module = importlib.util.module_from_spec(utils_module_spec)
        sys.modules["gomoku_rl.utils.module"] = utils_module_module
        utils_module_spec.loader.exec_module(utils_module_module)
        
        # Import utils.misc
        utils_misc_spec = importlib.util.spec_from_file_location(
            "gomoku_rl.utils.misc",
            os.path.join(gomoku_rl_path, "utils", "misc.py")
        )
        utils_misc_module = importlib.util.module_from_spec(utils_misc_spec)
        sys.modules["gomoku_rl.utils.misc"] = utils_misc_module
        utils_misc_spec.loader.exec_module(utils_misc_module)
        
        # Import utils.policy
        utils_policy_spec = importlib.util.spec_from_file_location(
            "gomoku_rl.utils.policy",
            os.path.join(gomoku_rl_path, "utils", "policy.py")
        )
        utils_policy_module = importlib.util.module_from_spec(utils_policy_spec)
        sys.modules["gomoku_rl.utils.policy"] = utils_policy_module
        utils_policy_spec.loader.exec_module(utils_policy_module)
        uniform_policy = utils_policy_module.uniform_policy
        
        # Import policy.common (needed by policy modules)
        policy_common_spec = importlib.util.spec_from_file_location(
            "gomoku_rl.policy.common",
            os.path.join(gomoku_rl_path, "policy", "common.py")
        )
        policy_common_module = importlib.util.module_from_spec(policy_common_spec)
        sys.modules["gomoku_rl.policy.common"] = policy_common_module
        policy_common_spec.loader.exec_module(policy_common_module)
        
        # Import policy base
        base_spec = importlib.util.spec_from_file_location(
            "gomoku_rl.policy.base",
            os.path.join(gomoku_rl_path, "policy", "base.py")
        )
        base_module = importlib.util.module_from_spec(base_spec)
        sys.modules["gomoku_rl.policy.base"] = base_module
        base_spec.loader.exec_module(base_module)
        
        # Import policy PPO and DQN (needed by __init__.py)
        ppo_spec = importlib.util.spec_from_file_location(
            "gomoku_rl.policy.ppo",
            os.path.join(gomoku_rl_path, "policy", "ppo.py")
        )
        ppo_module = importlib.util.module_from_spec(ppo_spec)
        sys.modules["gomoku_rl.policy.ppo"] = ppo_module
        ppo_spec.loader.exec_module(ppo_module)
        
        dqn_spec = importlib.util.spec_from_file_location(
            "gomoku_rl.policy.dqn",
            os.path.join(gomoku_rl_path, "policy", "dqn.py")
        )
        dqn_module = importlib.util.module_from_spec(dqn_spec)
        sys.modules["gomoku_rl.policy.dqn"] = dqn_module
        dqn_spec.loader.exec_module(dqn_module)
        
        # Now import policy __init__
        policy_spec = importlib.util.spec_from_file_location(
            "gomoku_rl.policy",
            os.path.join(gomoku_rl_path, "policy", "__init__.py")
        )
        policy_module = importlib.util.module_from_spec(policy_spec)
        sys.modules["gomoku_rl.policy"] = policy_module
        policy_spec.loader.exec_module(policy_module)
        get_policy = policy_module.get_policy
        Policy = policy_module.Policy
        
        # Import torchrl tensor specs (this might still fail, but we've avoided the env import)
        from torchrl.data.tensor_specs import (
            DiscreteTensorSpec,
            CompositeSpec,
            UnboundedContinuousTensorSpec,
            BinaryDiscreteTensorSpec,
        )
        
        # Import tensordict
        from tensordict import TensorDict
        
    except ImportError as e:
        # Mark as failed to prevent retries
        _imports_failed = True
        
        # Check if it's the specific tensordict.memmap error
        error_str = str(e)
        if "MemmapTensor" in error_str or "tensordict.memmap" in error_str:
            error_msg = (
                "\n" + "="*70 + "\n"
                "IMPORT ERROR: Compatibility issue between tensordict and torchrl\n"
                "="*70 + "\n\n"
                "The error 'cannot import name MemmapTensor from tensordict.memmap'\n"
                "indicates a version mismatch between tensordict and torchrl.\n\n"
                "SOLUTIONS:\n"
                "1. Update both packages:\n"
                "   pip install --upgrade tensordict torchrl\n\n"
                "2. Install compatible versions (example):\n"
                "   pip install tensordict>=0.2.0 torchrl>=0.3.0\n\n"
                "3. Check current versions:\n"
                "   pip show tensordict torchrl\n\n"
                "4. If using a requirements file, ensure versions are compatible.\n\n"
                "Original error: " + error_str + "\n"
                "="*70 + "\n"
            )
            _import_error_message = error_msg
            # Print only once to stderr
            print(error_msg, file=sys.stderr)
            logging.error("Import failed: " + error_str)
        else:
            error_msg = f"Failed to import required modules: {e}"
            _import_error_message = error_msg
            logging.error(error_msg)
        
        raise ImportError(_import_error_message) from e
    
    # Store in globals for use in functions
    globals()['get_policy'] = get_policy
    globals()['Policy'] = Policy
    globals()['uniform_policy'] = uniform_policy
    globals()['Gomoku'] = Gomoku
    globals()['compute_done'] = compute_done
    globals()['DiscreteTensorSpec'] = DiscreteTensorSpec
    globals()['CompositeSpec'] = CompositeSpec
    globals()['UnboundedContinuousTensorSpec'] = UnboundedContinuousTensorSpec
    globals()['BinaryDiscreteTensorSpec'] = BinaryDiscreteTensorSpec
    globals()['TensorDict'] = TensorDict
    
    _imports_loaded = True


def load_config():
    """Load the demo configuration."""
    global _config_cache
    if _config_cache is None:
        config_path = os.path.join(CONFIG_PATH, "demo.yaml")
        _config_cache = OmegaConf.load(config_path)
        
        # Handle Hydra defaults - merge referenced configs
        if "defaults" in _config_cache:
            defaults = _config_cache.pop("defaults")
            for default_item in defaults:
                # Skip _self_ which is a Hydra special value
                if default_item == "_self_":
                    continue
                    
                if isinstance(default_item, str):
                    # Handle items like "algo: ppo" -> load algo/ppo.yaml
                    if ":" in default_item:
                        group, name = default_item.split(":", 1)
                        group = group.strip()
                        name = name.strip()
                        # Load the referenced config
                        ref_config_path = os.path.join(CONFIG_PATH, group, f"{name}.yaml")
                        if os.path.exists(ref_config_path):
                            ref_config = OmegaConf.load(ref_config_path)
                            # Merge into main config under the group name
                            # Use OmegaConf.merge to properly merge the configs
                            if group not in _config_cache:
                                _config_cache[group] = ref_config
                            else:
                                _config_cache[group] = OmegaConf.merge(_config_cache[group], ref_config)
                        else:
                            logging.warning(f"Config file not found: {ref_config_path}")
                elif isinstance(default_item, dict):
                    # Handle dict format if needed
                    for group, name in default_item.items():
                        ref_config_path = os.path.join(CONFIG_PATH, group, f"{name}.yaml")
                        if os.path.exists(ref_config_path):
                            ref_config = OmegaConf.load(ref_config_path)
                            if group not in _config_cache:
                                _config_cache[group] = ref_config
                            else:
                                _config_cache[group] = OmegaConf.merge(_config_cache[group], ref_config)
                        else:
                            logging.warning(f"Config file not found: {ref_config_path}")
        
        # Ensure algo.name exists before resolving (needed for checkpoint path interpolation)
        if "algo" in _config_cache and "name" not in _config_cache.algo:
            # If algo exists but name is missing, set a default
            _config_cache.algo.name = "ppo"
        elif "algo" not in _config_cache:
            # If algo doesn't exist at all, create it with default
            _config_cache.algo = OmegaConf.create({"name": "ppo"})
        
        # Register resolver before resolving
        OmegaConf.register_new_resolver("eval", eval)
        
        # Now resolve interpolations - algo.name should be available now
        try:
            OmegaConf.resolve(_config_cache)
        except Exception as e:
            logging.warning(f"Error resolving config interpolations: {e}. Continuing with unresolved config.")
            # If resolution fails, we'll handle it in get_model
            pass
    
    return _config_cache


def _ensure_imports():
    """Ensure imports are loaded."""
    if not _imports_loaded:
        _load_imports()


def make_model(cfg: DictConfig):
    """Create a model from configuration."""
    _ensure_imports()
    board_size = cfg.board_size
    device = cfg.device
    action_spec = DiscreteTensorSpec(
        board_size * board_size,
        shape=[1],
        device=device,
    )
    # when using PPO, setting num_envs=1 will cause an error in critic
    observation_spec = CompositeSpec(
        {
            "observation": UnboundedContinuousTensorSpec(
                device=cfg.device,
                shape=[2, 3, board_size, board_size],
            ),
            "action_mask": BinaryDiscreteTensorSpec(
                n=board_size * board_size,
                device=device,
                shape=[2, board_size * board_size],
                dtype=torch.bool,
            ),
        },
        shape=[2],
        device=device,
    )
    model = get_policy(
        name=cfg.algo.name,
        cfg=cfg.algo,
        action_spec=action_spec,
        observation_spec=observation_spec,
        device=cfg.device,
    )
    return model


def get_model(board_size: int, checkpoint_path: str = None):
    """Get or create a model for the given board size."""
    cache_key = f"{board_size}_{checkpoint_path or 'default'}"
    
    if cache_key not in _model_cache:
        cfg = load_config()
        cfg.board_size = board_size
        
        # Ensure algo section exists - if not, create a default
        if "algo" not in cfg or "name" not in cfg.algo:
            logging.warning("algo.name not found in config, using default 'ppo'")
            if "algo" not in cfg:
                cfg.algo = OmegaConf.create()
            cfg.algo.name = "ppo"
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            model = make_model(cfg)
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
                model.eval()
                logging.info(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                logging.warning(f"Failed to load checkpoint {checkpoint_path}: {e}. Using uniform policy.")
                model = uniform_policy
        else:
            # Try default checkpoint path
            default_checkpoint = cfg.get("checkpoint", None)
            # If checkpoint path has unresolved interpolations, try to resolve them manually
            if default_checkpoint and isinstance(default_checkpoint, str) and "${" in default_checkpoint:
                try:
                    # Try to resolve the checkpoint path
                    temp_cfg = OmegaConf.create({"checkpoint": default_checkpoint})
                    OmegaConf.resolve(temp_cfg)
                    default_checkpoint = temp_cfg.checkpoint
                except Exception:
                    # If resolution fails, construct a default path
                    algo_name = cfg.get("algo", {}).get("name", "ppo")
                    default_checkpoint = f"pretrained_models/{board_size}_{board_size}/{algo_name}/0.pt"
                    logging.warning(f"Could not resolve checkpoint path, using: {default_checkpoint}")
            
            if default_checkpoint and os.path.exists(default_checkpoint):
                model = make_model(cfg)
                try:
                    model.load_state_dict(torch.load(default_checkpoint, map_location=cfg.device))
                    model.eval()
                    logging.info(f"Loaded default checkpoint from {default_checkpoint}")
                except Exception as e:
                    logging.warning(f"Failed to load default checkpoint: {e}. Using uniform policy.")
                    model = uniform_policy
            else:
                logging.info("No checkpoint found, using uniform policy")
                model = uniform_policy
        
        _model_cache[cache_key] = model
    
    return _model_cache[cache_key]


def parse_board_matrix(matrix: str, board_size: int):
    """Parse board matrix string into 2D list.
    
    Args:
        matrix: String representation of board (e.g., JSON array or comma-separated)
        board_size: Size of the board
        
    Returns:
        2D list of strings (board[row][col])
    """
    matrix = matrix.strip()
    
    # Try parsing as JSON array first
    try:
        parsed = json.loads(matrix)
        if isinstance(parsed, list) and len(parsed) > 0:
            if isinstance(parsed[0], list):
                # It's a 2D array
                board = []
                for row in parsed:
                    normalized_row = []
                    for cell in row:
                        cell_str = str(cell).upper() if cell else '-'
                        if cell_str in ['', ' ', '-', '_', '0', 'NULL', 'NONE', 'EMPTY']:
                            cell_str = '-'
                        normalized_row.append(cell_str)
                    board.append(normalized_row)
                
                # Validate dimensions
                if len(board) != board_size:
                    raise ValueError(f"Expected {board_size} rows, got {len(board)}")
                for i, row in enumerate(board):
                    if len(row) != board_size:
                        raise ValueError(f"Row {i} has {len(row)} columns, expected {board_size}")
                
                return board
            else:
                # It's a 1D array, convert to 2D
                board = []
                for i in range(board_size):
                    row = []
                    for j in range(board_size):
                        idx = i * board_size + j
                        if idx < len(parsed):
                            cell = str(parsed[idx]).upper()
                            if cell in ['', ' ', '-', '_', '0', 'NULL', 'NONE', 'EMPTY']:
                                cell = '-'
                        else:
                            cell = '-'
                        row.append(cell)
                    board.append(row)
                return board
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        pass
    
    # Try comma-separated format
    if ',' in matrix:
        cells = [c.strip().upper() for c in matrix.split(',')]
        board = []
        for i in range(board_size):
            row = []
            for j in range(board_size):
                idx = i * board_size + j
                if idx < len(cells):
                    cell = cells[idx]
                    if cell in ['', ' ', '-', '_', '0', 'NULL', 'NONE', 'EMPTY']:
                        cell = '-'
                else:
                    cell = '-'
                row.append(cell)
            board.append(row)
        return board
    
    # Default: empty board
    return [['-' for _ in range(board_size)] for _ in range(board_size)]


def set_gomoku_board_state(env, board_2d: list, next_player: str, last_move_row: int = None, last_move_col: int = None):
    """Set the Gomoku environment board state from a 2D board representation.
    
    Args:
        env: Gomoku environment instance
        board_2d: 2D list representing the board ('X', 'O', '-')
        next_player: Next player to move ('X' or 'O')
        last_move_row: Row of last move (optional)
        last_move_col: Column of last move (optional)
    """
    board_size = env.board_size
    device = env.device
    
    # Reset the environment
    env.reset()
    
    # Convert board to Gomoku format (0=empty, 1=black, -1=white)
    # In Gomoku: turn 0 = black (1), turn 1 = white (-1)
    # In API: 'X' = black, 'O' = white
    move_count = 0
    last_move_idx = -1
    
    for i in range(board_size):
        for j in range(board_size):
            cell = board_2d[i][j].upper()
            if cell == 'X':
                env.board[0, i, j] = 1  # black
                move_count += 1
                last_move_idx = i * board_size + j
            elif cell == 'O':
                env.board[0, i, j] = -1  # white
                move_count += 1
                last_move_idx = i * board_size + j
    
    env.move_count[0] = move_count
    
    # Determine turn based on next player
    # If next player is X (black), then current turn should be 0 (black)
    # If next player is O (white), then current turn should be 1 (white)
    if next_player.upper() == 'X':
        env.turn[0] = 0  # black's turn
    else:
        env.turn[0] = 1  # white's turn
    
    # Set last move if provided or use the last non-empty cell
    if last_move_row is not None and last_move_col is not None:
        if 0 <= last_move_row < board_size and 0 <= last_move_col < board_size:
            env.last_move[0] = last_move_row * board_size + last_move_col
        else:
            env.last_move[0] = last_move_idx if last_move_idx >= 0 else -1
    else:
        env.last_move[0] = last_move_idx if last_move_idx >= 0 else -1
    
    # Check if game is done
    if move_count > 0:
        _ensure_imports()
        piece = torch.where(env.turn == 0, 1, -1)
        board_one_side = (env.board == piece.unsqueeze(-1).unsqueeze(-1)).float()
        env.done[0] = compute_done(
            board_one_side,
            env.kernel_horizontal,
            env.kernel_vertical,
            env.kernel_diagonal,
        )[0] | (env.move_count[0] == board_size * board_size)
    else:
        env.done[0] = False


@app.route('/api/move', methods=['GET'])
def get_move():
    try:
        # Ensure imports are loaded before using them
        _ensure_imports()
        
        board_size = int(request.args.get('boardSize') or request.args.get('size') or '15')
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
                board_2d = parse_board_matrix(matrix, board_size)
            except Exception as e:
                return jsonify({'error': f'Invalid matrix format: {e}'}), 400
        else:
            board_2d = [['-' for _ in range(board_size)] for _ in range(board_size)]
        
        # Get checkpoint path from request (optional)
        checkpoint_path = request.args.get('checkpoint')
        
        # Get model
        model = get_model(board_size, checkpoint_path)
        
        # Create Gomoku environment
        device = "cpu"  # Use CPU for API
        env = Gomoku(num_envs=1, board_size=board_size, device=device)
        
        # Set board state
        set_gomoku_board_state(env, board_2d, next_move, last_move_row, last_move_col)
        
        # Check if game is already done
        if env.done.item():
            return jsonify({'error': 'Game is already finished'}), 400
        
        # Get the model's prediction
        tensordict = TensorDict(
            {
                "observation": env.get_encoded_board(),
                "action_mask": env.get_action_mask(),
            },
            batch_size=1,
        )
        
        with torch.no_grad():
            tensordict = model(tensordict).cpu()
        
        action: int = tensordict["action"].item()
        
        # Convert action to row/col
        # Note: In demo.py, action is converted as:
        # x = action // board_size
        # y = action % board_size
        # So action = x * board_size + y
        best_row = action // board_size
        best_col = action % board_size
        
        # Validate the move
        if not env.is_valid(torch.tensor([action])).item():
            # If invalid, find first valid move as fallback
            action_mask = env.get_action_mask()[0]
            valid_actions = torch.nonzero(action_mask, as_tuple=False).squeeze(-1)
            if len(valid_actions) > 0:
                action = valid_actions[0].item()
                best_row = action // board_size
                best_col = action % board_size
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
    port = int(os.getenv('API_PORT', 5050))
    host = os.getenv('API_HOST', '0.0.0.0')
    debug = os.getenv('API_DEBUG', 'False').lower() == 'true'
    
    logging.basicConfig(level=logging.INFO)
    print(f"Starting Flask API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
