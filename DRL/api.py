from __future__ import annotations

import json
import os
from typing import List

import numpy as np
import torch
from flask import Flask, jsonify, request
from ast import literal_eval

from checkpoint import CheckpointManager
from env import GomokuEnv
from model import PolicyValueNet
from utils import get_device, load_config

app = Flask(__name__)

CONFIG_PATH = os.environ.get("GOMOKU_CONFIG", "config.yaml")
CONFIG = load_config(CONFIG_PATH)
TRAINED_BOARD_SIZE = int(CONFIG["board_size"])
TRAINED_WIN_LENGTH = int(CONFIG["win_length"])
CHECKPOINT_DIR = CONFIG.get("checkpoint_dir", "checkpoints")
DEVICE = get_device(
    CONFIG.get("training", {}).get("device"),
    CONFIG.get("training", {}).get("max_vram_gb"),
)

MODEL: PolicyValueNet | None = None
CHECKPOINT_ERROR: str | None = None


def load_latest_checkpoint() -> None:
    global MODEL, CHECKPOINT_ERROR
    manager = CheckpointManager(CHECKPOINT_DIR, CONFIG["training"].get("max_checkpoints", 5))
    latest = manager.latest_checkpoint()
    if latest is None:
        CHECKPOINT_ERROR = "No checkpoint available. Train the agent first."
        return
    MODEL = PolicyValueNet(board_size=TRAINED_BOARD_SIZE).to(DEVICE)
    MODEL.eval()
    state = manager.load(str(latest))
    MODEL.load_state_dict(state["model_state"])
    print(f"Loaded checkpoint: {latest}")


def parse_board(board_param: str, board_size: int) -> List[List[int]]:
    try:
        parsed = json.loads(board_param)
    except json.JSONDecodeError:
        parsed = literal_eval(board_param)
    board = np.array(parsed)
    if board.shape != (board_size, board_size):
        raise ValueError("Board shape mismatch")
    normalize = np.vectorize(_normalize_cell)
    return normalize(board).tolist()


def _normalize_cell(value) -> int:
    if isinstance(value, str):
        val = value.strip().upper()
        if val in ("X", "1"):
            return 1
        if val in ("O", "-1"):
            return -1
        return 0
    if value == 1:
        return 1
    if value == -1:
        return -1
    return 0


def build_board(board_size: int, last_row: int | None, last_col: int | None, next_player: str) -> List[List[int]]:
    board = [[0 for _ in range(board_size)] for _ in range(board_size)]
    if last_row is not None and last_col is not None:
        prev_player = -1 if next_player == "X" else 1
        if 0 <= last_row < board_size and 0 <= last_col < board_size:
            board[last_row][last_col] = prev_player
    return board


@app.route("/api/move", methods=["GET"])
def get_move():
    if MODEL is None:
        load_latest_checkpoint()
    if MODEL is None or CHECKPOINT_ERROR:
        return jsonify({"error": CHECKPOINT_ERROR}), 500

    try:
        board_size = int(request.args.get("boardSize") or request.args.get("size") or TRAINED_BOARD_SIZE)
        win_length = int(request.args.get("winLength") or request.args.get("win") or TRAINED_WIN_LENGTH)
        next_move = (request.args.get("nextMove") or request.args.get("player") or "X").upper()
        last_row = request.args.get("last_move_row")
        last_col = request.args.get("last_move_col")
        board_param = request.args.get("board")

        if board_param:
            board = parse_board(board_param, board_size)
        else:
            board = build_board(
                board_size,
                int(last_row) if last_row is not None else None,
                int(last_col) if last_col is not None else None,
                next_move,
            )

        env = GomokuEnv(board_size, win_length)
        current_player = 1 if next_move == "X" else -1
        observation = env.set_state(board, current_player)
        mask = env.legal_actions_mask()
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=DEVICE)
        MODEL.board_size = board_size

        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, _ = MODEL(obs_tensor)
            logits = logits.squeeze(0)
            logits = logits.masked_fill(~mask_tensor, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.argmax(probs).item())
            confidence = float(probs[action].item())

        row = action // board_size
        col = action % board_size
        response = {
            "row": row,
            "col": col,
            "action": action,
            "confidence": confidence,
            "player": next_move,
            "boardSize": board_size,
            "winLength": win_length,
        }
        return jsonify(response)
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    load_latest_checkpoint()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)))
