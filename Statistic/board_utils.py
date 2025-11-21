import requests
from typing import List, Tuple, Optional, Dict, Callable
import json

# Global execute_query function (will be set by api_server.py)
execute_query: Callable[[str], int] = None

def set_execute_query(func: Callable[[str], int]):
    """
    Set the execute_query function to be used by query functions
    
    Args:
        func: Function that takes SQL string and returns count (int)
    """
    global execute_query
    execute_query = func

def parse_board_matrix(matrix: str, board_size: int) -> List[List[str]]:
    """
    Parse board matrix string into 2D list
    
    Args:
        matrix: String representation of board (e.g., "X-O-X-O-..." or "X,O,-,X,O,..." or JSON array)
        board_size: Size of the board
        
    Returns:
        2D list of strings (board[row][col])
    """
    matrix = matrix.strip()
    
    # Try parsing as JSON array first (e.g., [["X","-","-"],["-","-","-"],...])
    try:
        parsed = json.loads(matrix)
        if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
            # It's a 2D array
            board = []
            for row in parsed:
                normalized_row = []
                for cell in row:
                    cell_str = str(cell).upper() if cell else '-'
                    if cell_str in ['', ' ', '-', '_', '0', 'NULL', 'NONE']:
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
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        # Not JSON, continue with other parsing methods
        pass
    
    # Remove whitespace
    matrix_clean = matrix.replace(' ', '')
    
    # Try comma-separated first
    if ',' in matrix_clean:
        # Remove brackets and quotes if present
        matrix_clean = matrix_clean.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
        cells = [c.strip() for c in matrix_clean.split(',') if c.strip()]
    # Try newline-separated
    elif '\n' in matrix:
        lines = matrix.split('\n')
        cells = []
        for line in lines:
            cells.extend([c for c in line.strip() if c and c not in ['[', ']', '"', "'"]])
    # Otherwise, treat as single string (remove brackets/quotes)
    else:
        matrix_clean = matrix_clean.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
        cells = [c for c in matrix_clean if c and c not in [',', ' ', '\n', '\r', '\t']]
    
    # Validate length
    expected_length = board_size * board_size
    if len(cells) != expected_length:
        raise ValueError(f"Expected {expected_length} cells, got {len(cells)}. Matrix preview: {matrix[:200]}")
    
    # Convert to 2D list
    board = []
    for i in range(board_size):
        row = []
        for j in range(board_size):
            idx = i * board_size + j
            cell = cells[idx].upper() if idx < len(cells) else '-'
            # Normalize: empty, space, or dash becomes '-'
            if cell in ['', ' ', '-', '_', '0', 'NULL', 'NONE']:
                cell = '-'
            row.append(cell)
        board.append(row)
    
    return board


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
        sql = f"SELECT COUNT(win_actor) FROM {table_name} WHERE {where_clause}"
        
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
    for level in range(10, 25, 2):  # 10, 12, 14, ..., 24
        if level < move_count:
            continue
        
        table_name = f"ttt_5_l{level}"
        sql = f"SELECT COUNT(win_actor) FROM {table_name} WHERE {where_clause}"
        
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
    sql = f"SELECT COUNT(win_actor) FROM ttt_5_draw WHERE {where_clause}"
    
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