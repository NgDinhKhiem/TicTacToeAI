import requests
import numpy as np
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

execute_query: Callable[[str], int] = None

def set_execute_query(func: Callable[[str], int]):
    """
    Set the execute_query function to be used by query functions
    
    Args:
        func: Function that takes SQL string and returns count (int)
    """
    global execute_query
    execute_query = func

def get_odd_table_names(move_count: int) -> str:
    """
    Lấy danh sách tên bảng odd (lượt lẻ)
    
    Args:
        move_count: Số nước đã đi (không dùng, chỉ để giữ signature)
        
    Returns:
        String format SQL: ttt_5_l9, ttt_5_l11, ..., ttt_5_l25
    """
    tables = []
    
    # Lấy tất cả các level lẻ từ 9 đến 25
    for level in range(9, 26, 2):  # 9, 11, 13, ..., 25
        tables.append(f"ttt_5_l{level}")
    
    return ", ".join(tables)


def get_even_table_names(move_count: int) -> str:
    """
    Lấy danh sách tên bảng even (lượt chẵn)
    
    Args:
        move_count: Số nước đã đi (không dùng, chỉ để giữ signature)
        
    Returns:
        String format SQL: ttt_5_l10, ttt_5_l12, ..., ttt_5_l24
    """
    tables = []
    
    # Lấy tất cả các level chẵn từ 10 đến 24
    for level in range(10, 25, 2):  # 10, 12, 14, ..., 24
        tables.append(f"ttt_5_l{level}")
    
    return ", ".join(tables)


def build_where_clause(board: list) -> str:
    """
    Xây dựng WHERE clause từ board
    
    Args:
        board: Board hiện tại
        
    Returns:
        WHERE clause string
    """
    n = 5
    conditions = []
    
    for idx, cell in enumerate(board):
        if cell != 0:
            row = (idx // n) + 1  # +1 vì index bắt đầu từ 1
            col = (idx % n) + 1
            col_name = f"i{row}{col}"
            player_mark = 'X' if cell == 1 else 'O'
            conditions.append(f"{col_name} = '{player_mark}'")
    
    return " AND ".join(conditions) if conditions else "1=1"


def query_odd_table(board: list) -> int:
    """
    Query bảng odd (lượt lẻ) - đếm số trận X thắng
    
    Args:
        board: Bảng hiện tại (list of int, size 25)
        
    Returns:
        Số lượng rows có win_actor = 'X'
    """
    move_count = sum(1 for cell in board if cell != 0)
    
    if move_count == 0:
        return 0
    
    where_clause = build_where_clause(board)
    
    # Đếm trực tiếp từng bảng và cộng lại
    total_count = 0
    for level in range(9, 26, 2):  # 9, 11, 13, ..., 25
        if level < move_count:
            continue
        
        table_name = f"ttt_5_l{level}"
        sql = f"SELECT COUNT(win_actor) FROM {table_name} WHERE {where_clause} AND win_actor = 'X'"
        
        count = execute_query(sql)
        total_count += count
    
    return total_count


def query_even_table(board: list) -> int:
    """
    Query bảng even (lượt chẵn) - đếm số trận O thắng
    
    Args:
        board: Bảng hiện tại (list of int, size 25)
        
    Returns:
        Số lượng rows có win_actor = 'O'
    """
    move_count = sum(1 for cell in board if cell != 0)
    
    if move_count == 0:
        return 0
    
    where_clause = build_where_clause(board)
    
    # Đếm trực tiếp từng bảng và cộng lại
    total_count = 0
    for level in range(10, min(25, max(10, total_count+7)), 2):  # 10, 12, 14, ..., 24
        if level < move_count:
            continue
        
        table_name = f"ttt_5_l{level}"
        sql = f"SELECT COUNT(win_actor) FROM {table_name} WHERE {where_clause} AND win_actor = 'O'"
        
        count = execute_query(sql)
        total_count += count
    
    return total_count


def query_draw_table(board: list) -> int:
    """
    Query bảng draw (ttt_5_draw) - đếm số trận hòa
    
    Args:
        board: Bảng hiện tại (list of int, size 25)
        
    Returns:
        Số lượng rows có win_actor = 'D'
    """
    move_count = sum(1 for cell in board if cell != 0)
    
    if move_count == 0:
        return 0
    
    where_clause = build_where_clause(board)
    
    # Query table ttt_5_draw
    sql = f"SELECT COUNT(win_actor) FROM ttt_5_draw WHERE {where_clause} AND win_actor = 'D'"
    
    return execute_query(sql)

#=========================================Symmetric==========================================
N = 5  # Board size constant

# Transformation functions
def t_identity(r, c):
    return (r, c)

def t_rot90(r, c):
    return (c, N-1-r)

def t_rot180(r, c):
    return (N-1-r, N-1-c)

def t_rot270(r, c):
    return (N-1-c, r)

def t_reflect_h(r, c):
    return (N-1-r, c)

def t_reflect_v(r, c):
    return (r, N-1-c)

def t_reflect_main(r, c):
    return (c, r)

def t_reflect_anti(r, c):
    return (N-1-c, N-1-r)


def apply_transformation(board: list, transform_func) -> list:
    """
    Áp dụng transformation function lên board
    
    Args:
        board: Board 1D (25 elements)
        transform_func: Hàm transformation (r,c) -> (r',c')
        
    Returns:
        Board mới sau khi transform
    """
    n = N
    new_board = [0] * (n * n)
    
    for idx in range(n * n):
        r = idx // n
        c = idx % n
        
        # Apply transformation
        new_r, new_c = transform_func(r, c)
        new_idx = new_r * n + new_c
        
        new_board[new_idx] = board[idx]
    
    return new_board


def get_symmetries(board: list) -> list:
    """
    Tạo tất cả các phép biến đổi đối xứng của board 5x5
    Dùng cùng transformations như lúc gen data
    
    Args:
        board: Board hiện tại (list 25 elements)
        
    Returns:
        List các board đối xứng (8 biến đổi)
    """
    transformations = [
        t_identity,
        t_rot90,
        t_rot180,
        t_rot270,
        t_reflect_h,
        t_reflect_v,
        t_reflect_main,
        t_reflect_anti
    ]
    
    symmetries = []
    for transform in transformations:
        sym_board = apply_transformation(board, transform)
        symmetries.append(sym_board)
    
    return symmetries


def canonical_board(board: list) -> list:
    """
    Tìm canonical form của board (form nhỏ nhất theo lexicographic order)
    Giống như lúc gen data
    
    Args:
        board: Board hiện tại
        
    Returns:
        Canonical board
    """
    symmetries = get_symmetries(board)
    
    # Convert to tuples for comparison
    sym_tuples = [tuple(sym) for sym in symmetries]
    
    # Return the lexicographically smallest
    return list(min(sym_tuples))

#=========================================5x5 Logic==========================================
def get_steps_with_rate(currBoard: list[list[int]], player: int, glob_r: int, glob_c: int) -> list[list[list[int]]]:
    """
    Tìm nước đi tốt nhất cho AI dựa trên database
    
    Args:
        currBoard: Board hiện tại (2D array 5x5)
        player: Player hiện tại (1 hoặc 2)
        
    Returns:
        3D array [5][5][3] với [win_rate, lose_rate, draw_rate]
    """
    # Log số ô trống
    empty_cells = sum(1 for cell in currBoard if cell == 0)
    print(f"\n🤔 AI đang suy nghĩ... (Còn {empty_cells} ô trống)")

    steps_with_rate = [[[] for _ in range(5)] for _ in range(5)]

    for c in range(5):
        for r in range(5):
            if currBoard[r][c] != 0:
                continue
            
            # Tìm canonical form

            newBoard = copy.deepcopy(currBoard)
            newBoard[r][c] = player
            board_1d = convert_to_db_schema_1d(newBoard)
            canonical = canonical_board(board_1d)
            
            # Query với canonical form
            x_win_count = query_odd_table(canonical)
            o_win_count = query_even_table(canonical)
            draw_count = query_draw_table(canonical)
            
            total_count = x_win_count + o_win_count + draw_count
            
            if total_count <= 0:
                steps_with_rate[r][c] = [0.0, 0.0, 0.0, 0.0]
                continue
            
            # Tính win rate và lose rate cho player hiện tại
            win_count = x_win_count if player == 1 else o_win_count
            lose_count = o_win_count if player == 1 else x_win_count

            steps_with_rate[r][c] = [win_count, lose_count, draw_count, total_count]

            # Tính win rate và lose rate cho player hiện tại
            current_win_rate = win_count / total_count
            current_lose_rate = lose_count / total_count
            draw_rate = draw_count / total_count
            
            # Log chi tiết
            print(f"  Ô [{r + glob_c - 2},{c + glob_r - 2}]): "
                f"win={current_win_rate:.2%}, lose={current_lose_rate:.2%}, draw={draw_rate:.2%} "
                f"(X:{x_win_count}, O:{o_win_count}, D:{draw_count}, total:{total_count})")
    
    return steps_with_rate

#===================================Unlimited Space logic====================================
BOARD_SIZE = 15
LOSE_THRESHOLD = 0.05 # 5%

def best_steps_unlimited(currBoard: list[list[int]], player: int, last_move_col: int, last_move_row: int):
    """
    Tìm nước đi tốt nhất cho AI trong unlimited space
    
    Args:
        currBoard: Board hiện tại (100x100)
        player: Player hiện tại (1 hoặc 2)

    Logic:
        1. xác định không gian 5x5 quanh bước đi cuối cùng
        2. tìm tất cả nước đi của player trong khdrông gian 5x5 đó
        3. tìm best_step của mỗi bước với không gian 5x5 mới bao quanh bước đi đó
        4. sum win_rate của các bước đi, chọn bước có tổng win_rate cao nhất
        
    Returns:
        Index của nước đi tốt nhất, hoặc -1 nếu không tìm thấy
    """

    board_5_x_5 = []
    col_min = 0
    col_max = 0
    row_min = 0
    row_max = 0

    # Adjust for board edges
    col_index = get_col_index_5_x_5(last_move_col)
    row_index = get_row_index_5_x_5(last_move_row)

    # Identify 5x5 checking area
    col_min = last_move_col - 2 + col_index
    col_max = last_move_col + 2 + col_index
    row_min = last_move_row - 2 + row_index
    row_max = last_move_row + 2 + row_index

    # Accumulate win/lose rates cho mỗi ô trống
    board_accumulated = {}
    opponent = 2 if player == 1 else 1
    
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            if currBoard[r][c] != opponent:
                continue
            
            # Lấy board 5x5 xung quanh vị trí player này (dạng 2D)
            board_1d = get_board_5_x_5(currBoard, r, c)
            board_2d = board_1d_to_2d(board_1d)


            print ("AI thinking for board with center at ({}, {})".format(r, c))
            steps_with_rate = get_steps_with_rate(board_2d, player, r, c)
            
            # Map local 5x5 coordinates về global 100x100
            p_col_index = get_col_index_5_x_5(c)
            p_row_index = get_row_index_5_x_5(r)
            
            for local_r in range(5):
                for local_c in range(5):
                    global_r = r - 2 + p_row_index + local_r
                    global_c = c - 2 + p_col_index + local_c
                    
                    # Chỉ accumulate nếu ô trống
                    if (global_r >= 0 and global_r < BOARD_SIZE and
                        global_c >= 0 and global_c < BOARD_SIZE and
                        currBoard[global_r][global_c] == 0):

                        rates = steps_with_rate[local_r][local_c]                        
                        if (global_r, global_c) not in board_accumulated:
                            board_accumulated[(global_r, global_c)] = [0.0, 0.0, 0.0, 0.0] # win_count, lose_count, draw_count, total_count

                        win_rate = rates[0] / rates[3]
                        lose_rate = rates[1] / rates[3]
                        draw_rate = rates[2] / rates[3] 


                        if win_rate > board_accumulated[(global_r, global_c)][0]:
                            board_accumulated[(global_r, global_c)][0] = win_rate
                        if lose_rate > board_accumulated[(global_r, global_c)][1]:
                            board_accumulated[(global_r, global_c)][1] = lose_rate
                        if draw_rate > board_accumulated[(global_r, global_c)][2]:
                            board_accumulated[(global_r, global_c)][2] = draw_rate
                        


                            

    # ✅ Kiểm tra nếu không có data
    if not board_accumulated:
        return (-1, -1)

    best_column = -1
    best_row = -1
    highest_win_rate = -1.0
    lowest_lose_rate = 101.0

    # ✅ Chỉ duyệt qua các ô có data trong dictionary
    for (r, c), rates in board_accumulated.items():
        win_rate = rates[0]
        lose_rate = rates[1]
        print(f"({r}, {c}) win_rate: {win_rate}, lose_rate: {lose_rate}")
        if win_rate >= 99:
            return (r, c)
        if lose_rate < lowest_lose_rate:
            lowest_lose_rate = lose_rate
            best_column = c
            best_row = r
        if abs(lose_rate - lowest_lose_rate) < LOSE_THRESHOLD:
            if win_rate > highest_win_rate:
                highest_win_rate = win_rate
                best_column = c
                best_row = r

    print(f"best_row: {best_row}, best_column: {best_column}, highest_win_rate: {highest_win_rate}, lowest_lose_rate: {lowest_lose_rate}")
    return (best_row, best_column)

#=========================================Conversion Functions==========================================
def board_2d_to_1d(board_2d: list[list[int]]) -> list[int]:
    """
    Convert board 2D (5x5) sang 1D (25 elements)
    
    Args:
        board_2d: Board 2D array [[...], [...], ...]
        
    Returns:
        Board 1D array [...]
    """
    return [cell for row in board_2d for cell in row]


def board_1d_to_2d(board_1d: list[int]) -> list[list[int]]:
    """
    Convert board 1D (25 elements) sang 2D (5x5)
    
    Args:
        board_1d: Board 1D array [...]
        
    Returns:
        Board 2D array [[...], [...], ...]
    """
    return [board_1d[i:i+5] for i in range(0, 25, 5)]

def convert_to_db_schema_1d(board_1d: list[int]) -> list[int]:
    """
    Chuyển đổi board 1D từ get_board_5_x_5() sang format DB schema
    
    get_board_5_x_5() đang duyệt theo COL trước ROW (SAI):
        for c in range(...):
            for r in range(...):
                board_5_x_5_index = (r - ...) * 5 + (c - ...)
    
    DB Schema cần duyệt theo ROW trước COL (ĐÚNG):
        i11, i12, i13, i14, i15, i21, i22, ..., i55
    
    Args:
        board_1d: Board 1D [0-24] từ get_board_5_x_5() (col-major order)
        
    Returns:
        Board 1D [0-24] theo DB schema (row-major order: i11->i55)
    """
    result = [0] * 25
    for idx in range(25):
        # get_board_5_x_5 duyệt: col trước, row sau
        col = idx // 5
        row = idx % 5
        # DB schema cần: row trước, col sau
        db_index = row * 5 + col
        result[db_index] = board_1d[idx]
    return result

def convert_to_db_schema_1d(board_2d: list[list[int]]) -> list[int]:
    """
    Chuyển đổi board 2D sang format DB schema 1D
    
    DB Schema cần duyệt theo ROW trước COL:
        i11, i12, i13, i14, i15, i21, i22, ..., i55
    
    Args:
        board_2d: Board 2D array [[...], [...], ...] (5x5)
                  board_2d[row][col] với row, col từ 0 đến 4
        
    Returns:
        Board 1D [0-24] theo DB schema (row-major order: i11->i55)
        
    Example:
        board_2d = [[1, 0, 2, 0, 1],    # row 0: i11, i12, i13, i14, i15
                    [2, 1, 0, 1, 0],    # row 1: i21, i22, i23, i24, i25
                    [0, 0, 1, 0, 2],    # row 2: i31, i32, i33, i34, i35
                    [1, 2, 0, 2, 0],    # row 3: i41, i42, i43, i44, i45
                    [0, 1, 2, 0, 1]]    # row 4: i51, i52, i53, i54, i55
        
        result = [1, 0, 2, 0, 1, 2, 1, 0, 1, 0, 0, 0, 1, 0, 2, 1, 2, 0, 2, 0, 0, 1, 2, 0, 1]
    """
    result = []
    for row in range(5):
        for col in range(5):
            result.append(board_2d[row][col])
    return result

#==========================================Support===========================================
def get_board_5_x_5(currBoard: list[list[int]], center_row: int, center_col: int) -> list:
    col_index = get_col_index_5_x_5(center_col)
    row_index = get_row_index_5_x_5(center_row)
    result = [0] * 25

    for c in range(center_col - 2 + col_index, center_col + 2 + col_index + 1):
        for r in range(center_row - 2 + row_index, center_row + 2 + row_index + 1):
            board_5_x_5_index = (r - (center_row - 2 + row_index)) * 5 + (c - (center_col - 2 + col_index))
            result[board_5_x_5_index] = currBoard[r][c]

    return result

def get_col_index_5_x_5(last_move_col: int) -> int:
    # col limit (board edge)
    col_index = 0
    if last_move_col > BOARD_SIZE - 3:
        col_index = BOARD_SIZE - 3
    elif last_move_col < 2:
        col_index = 2

    return col_index
    
def get_row_index_5_x_5(last_move_row: int) -> int:
    # row limit (board edge)
    row_index = 0
    if last_move_row > BOARD_SIZE - 3:
        row_index = BOARD_SIZE - 3
    elif last_move_row < 2:
        row_index = 2

    return row_index

#==========================================Game Logic 15x15==========================================

class TicTacToe15x15:
    """
    Game logic cho Tic-Tac-Toe 15x15
    Chiến thắng khi có 5 ô liên tiếp theo hàng ngang, dọc hoặc chéo
    """
    
    def __init__(self):
        """Khởi tạo board 15x15"""
        self.board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 1  # Player 1 (X) đi trước
        self.last_move = None  # (row, col)
        self.winner = None
        self.game_over = False
        
    def reset(self):
        """Reset game về trạng thái ban đầu"""
        self.board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 1
        self.last_move = None
        self.winner = None
        self.game_over = False
        
    def is_valid_move(self, row: int, col: int) -> bool:
        """
        Kiểm tra nước đi có hợp lệ không
        
        Args:
            row: Hàng (0-14)
            col: Cột (0-14)
            
        Returns:
            True nếu hợp lệ, False nếu không
        """
        if self.game_over:
            return False
            
        if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            return False
            
        return self.board[row][col] == 0
    
    def make_move(self, row: int, col: int) -> bool:
        """
        Thực hiện nước đi
        
        Args:
            row: Hàng (0-14)
            col: Cột (0-14)
            
        Returns:
            True nếu thành công, False nếu không hợp lệ
        """
        if not self.is_valid_move(row, col):
            return False
            
        self.board[row][col] = self.current_player
        self.last_move = (row, col)
        
        # Kiểm tra thắng
        if self.check_winner(row, col):
            self.winner = self.current_player
            self.game_over = True
            return True
            
        # Kiểm tra hòa (board đầy)
        if self.is_board_full():
            self.game_over = True
            self.winner = 0  # Draw
            return True
            
        # Chuyển lượt
        self.current_player = 2 if self.current_player == 1 else 1
        return True
    
    def check_winner(self, row: int, col: int) -> bool:
        """
        Kiểm tra xem có người thắng không sau nước đi vừa rồi
        
        Args:
            row: Hàng vừa đi
            col: Cột vừa đi
            
        Returns:
            True nếu có người thắng
        """
        player = self.board[row][col]
        
        # Kiểm tra 4 hướng: ngang, dọc, chéo chính, chéo phụ
        directions = [
            (0, 1),   # Ngang
            (1, 0),   # Dọc
            (1, 1),   # Chéo chính
            (1, -1)   # Chéo phụ
        ]
        
        for dr, dc in directions:
            count = 1  # Đếm ô hiện tại
            
            # Đếm theo hướng thuận
            r, c = row + dr, col + dc
            while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                   self.board[r][c] == player):
                count += 1
                r += dr
                c += dc
            
            # Đếm theo hướng ngược
            r, c = row - dr, col - dc
            while (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and 
                   self.board[r][c] == player):
                count += 1
                r -= dr
                c -= dc
            
            # Nếu có 5 ô liên tiếp -> thắng
            if count >= 5:
                return True
                
        return False
    
    def is_board_full(self) -> bool:
        """Kiểm tra board đã đầy chưa"""
        for row in self.board:
            if 0 in row:
                return False
        return True
    
    def get_ai_move(self) -> tuple:
        """
        Lấy nước đi tốt nhất cho AI
        
        Returns:
            (row, col) hoặc (-1, -1) nếu không tìm thấy
        """
        if self.last_move is None:
            # Nước đi đầu tiên -> đi giữa board
            return (BOARD_SIZE // 2, BOARD_SIZE // 2)
        
        last_row, last_col = self.last_move
        return best_steps_unlimited(self.board, self.current_player, last_col, last_row)
    
    def print_board(self):
        """In board ra console (hiển thị đủ 15x15)"""
        print("\n    ", end="")
        # Header cột
        for i in range(BOARD_SIZE):
            print(f"{i:2d}", end=" ")
        print()
        print("   " + "─" * (BOARD_SIZE * 3 + 1))
        
        # In từng hàng
        for i, row in enumerate(self.board):
            print(f"{i:2d} │", end="")
            for cell in row:
                if cell == 0:
                    print(" ·", end=" ")
                elif cell == 1:
                    print(" X", end=" ")
                else:
                    print(" O", end=" ")
            print("│")
        
        print("   " + "─" * (BOARD_SIZE * 3 + 1))
        print()


def play_game_human_vs_ai():
    """
    Chơi game: Human (X) vs AI (O)
    """
    game = TicTacToe15x15()
    
    print("=" * 70)
    print(" " * 20 + "🎮 TIC-TAC-TOE 15x15")
    print(" " * 20 + "HUMAN vs AI")
    print("=" * 70)
    print("📋 Quy tắc: 5 ô liên tiếp theo hàng ngang/dọc/chéo để thắng")
    print("👤 Bạn là X (đi trước) | 🤖 AI là O")
    print("=" * 70)
    
    while not game.game_over:
        game.print_board()
        
        if game.current_player == 1:
            # Human turn
            print(f"🎯 Lượt của BẠN (X)")
            try:
                row = int(input("   Nhập hàng (0-14): "))
                col = int(input("   Nhập cột (0-14): "))
                
                if not game.make_move(row, col):
                    print("❌ Nước đi không hợp lệ! Thử lại.\n")
                    continue
                    
            except ValueError:
                print("❌ Vui lòng nhập số!\n")
                continue
            except KeyboardInterrupt:
                print("\n\n👋 Thoát game!")
                return
                
        else:
            # AI turn
            print(f"🤖 Lượt của AI (O)")
            start_time = time.time()
            row, col = game.get_ai_move()
            elapsed = time.time() - start_time
            
            if row == -1 or col == -1:
                print("⚠️  AI không tìm thấy nước đi tốt, chọn ngẫu nhiên...")
                # Tìm ô trống đầu tiên
                found = False
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        if game.is_valid_move(r, c):
                            row, col = r, c
                            found = True
                            break
                    if found:
                        break
            
            print(f"   AI đi: ({row}, {col}) - Suy nghĩ trong {elapsed:.2f}s")
            game.make_move(row, col)
            time.sleep(0.3)
    
    # Game kết thúc
    game.print_board()
    print("=" * 70)
    if game.winner == 0:
        print(" " * 30 + "🤝 HÒA!")
    elif game.winner == 1:
        print(" " * 25 + "🎉 CHÚC MỪNG! BẠN THẮNG!")
    else:
        print(" " * 28 + "💻 AI THẮNG!")
    print("=" * 70)


def play_game_ai_vs_ai():
    """
    Chơi game: AI (X) vs AI (O)
    """
    game = TicTacToe15x15()
    
    print("=" * 70)
    print(" " * 20 + "🤖 TIC-TAC-TOE 15x15")
    print(" " * 23 + "AI vs AI")
    print("=" * 70)
    print()
    
    move_count = 0
    
    while not game.game_over:
        game.print_board()
        
        player_name = "AI-X" if game.current_player == 1 else "AI-O"
        symbol = "X" if game.current_player == 1 else "O"
        print(f"🤖 Lượt của {player_name} ({symbol})")
        
        start_time = time.time()
        row, col = game.get_ai_move()
        elapsed = time.time() - start_time
        
        if row == -1 or col == -1:
            print(f"⚠️  {player_name} không tìm thấy nước đi tốt, chọn ngẫu nhiên...")
            # Tìm ô trống đầu tiên
            found = False
            for r in range(BOARD_SIZE):
                for c in range(BOARD_SIZE):
                    if game.is_valid_move(r, c):
                        row, col = r, c
                        found = True
                        break
                if found:
                    break
        
        print(f"   {player_name} đi: ({row}, {col}) - Suy nghĩ trong {elapsed:.2f}s")
        game.make_move(row, col)
        move_count += 1
        
        time.sleep(0.3)
    
    # Game kết thúc
    game.print_board()
    print("=" * 70)
    print(f"📊 Tổng số nước đi: {move_count}")
    if game.winner == 0:
        print(" " * 30 + "🤝 HÒA!")
    elif game.winner == 1:
        print(" " * 28 + "🎉 AI-X THẮNG!")
    else:
        print(" " * 28 + "💻 AI-O THẮNG!")
    print("=" * 70)


#==========================================Main==========================================