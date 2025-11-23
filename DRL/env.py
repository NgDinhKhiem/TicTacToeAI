from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Optional[int]]


class GomokuEnv:
    """Self-play Gomoku environment supporting dynamic board and win length."""

    def __init__(self, board_size: int, win_length: int) -> None:
        if win_length > board_size:
            raise ValueError("win_length cannot exceed board_size")
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1 for X, -1 for O

    # ------------------------------------------------------------------
    # Core environment API
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.board.fill(0)
        self.current_player = 1
        return self._get_observation()

    def step(self, action: int) -> StepResult:
        row, col = self.action_to_coord(action)
        reward = 0.0
        done = False
        winner: Optional[int] = None
        acting_player = self.current_player

        if self.board[row, col] != 0:
            # Illegal move loses immediately
            reward = -1.0
            done = True
            winner = -acting_player
        else:
            self.board[row, col] = acting_player
            winner = self._check_winner(row, col)
            if winner == acting_player:
                done = True
                reward = 1.0
            elif self.is_draw():
                done = True
                reward = 0.0
                winner = 0
            else:
                reward = 0.0

        if not done:
            self.current_player *= -1

        observation = self._get_observation()
        return StepResult(observation, reward, done, {"winner": winner})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        current = (self.board == self.current_player).astype(np.float32)
        opponent = (self.board == -self.current_player).astype(np.float32)
        turn_plane = np.full_like(current, 1.0 if self.current_player == 1 else 0.0)
        stacked = np.stack([current, opponent, turn_plane], axis=0)
        return stacked

    def is_draw(self) -> bool:
        return not (self.board == 0).any()

    def action_to_coord(self, action: int) -> Tuple[int, int]:
        row = action // self.board_size
        col = action % self.board_size
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError("Action index out of range")
        return row, col

    def coord_to_action(self, row: int, col: int) -> int:
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError("Coordinates out of bounds")
        return row * self.board_size + col

    def legal_actions_mask(self) -> np.ndarray:
        mask = (self.board.reshape(-1) == 0).astype(np.float32)
        return mask

    def render(self) -> str:
        mapping = {0: ".", 1: "X", -1: "O"}
        rows = [" ".join(mapping[val] for val in row) for row in self.board]
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # State utilities
    # ------------------------------------------------------------------
    def set_state(self, board: List[List[int]], current_player: int) -> np.ndarray:
        array = np.array(board, dtype=np.int8)
        if array.shape != (self.board_size, self.board_size):
            raise ValueError("Board shape mismatch")
        if not np.isin(array, [-1, 0, 1]).all():
            raise ValueError("Board values must be in {-1,0,1}")
        self.board = array
        self.current_player = 1 if current_player >= 0 else -1
        return self._get_observation()

    def copy_board(self) -> np.ndarray:
        return self.board.copy()

    def _check_winner(self, recent_row: int, recent_col: int) -> Optional[int]:
        player = self.board[recent_row, recent_col]
        if player == 0:
            return None
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            if self._count_in_direction(recent_row, recent_col, dr, dc, player) >= self.win_length:
                return player
        return None

    def _count_in_direction(self, row: int, col: int, dr: int, dc: int, player: int) -> int:
        count = 1  # include the origin
        for sign in (1, -1):
            r, c = row, col
            while True:
                r += dr * sign
                c += dc * sign
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    break
                if self.board[r, c] != player:
                    break
                count += 1
        return count


def board_from_string(board_str: str, board_size: int) -> List[List[int]]:
    """Utility to parse board from simple string notation."""
    rows = board_str.strip().split("/")
    if len(rows) != board_size:
        raise ValueError("Invalid board string")
    mapping = {"X": 1, "O": -1, "0": 0, ".": 0}
    parsed: List[List[int]] = []
    for row in rows:
        if len(row) != board_size:
            raise ValueError("Row length mismatch")
        parsed.append([mapping.get(ch.upper(), 0) for ch in row])
    return parsed
