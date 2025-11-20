from flask import Flask, request, jsonify
from clickhouse_driver import Client
import copy
import os
import time
import math
import json
import requests
from typing import List, Tuple, Optional, Dict, Callable

# ----------------------------
# Configuration (env or config.py)
# ----------------------------
try:
    from config import (
        CLICKHOUSE_HOST, CLICKHOUSE_PORT, CLICKHOUSE_USER,
        CLICKHOUSE_PASSWORD, CLICKHOUSE_DATABASE, API_HOST, API_PORT, API_DEBUG
    )
except Exception:
    CLICKHOUSE_HOST = os.getenv('CLICKHOUSE_HOST', 'localhost')
    CLICKHOUSE_PORT = int(os.getenv('CLICKHOUSE_PORT', 9000))
    CLICKHOUSE_USER = os.getenv('CLICKHOUSE_USER', 'admin')
    CLICKHOUSE_PASSWORD = os.getenv('CLICKHOUSE_PASSWORD', 'Helloworld')
    CLICKHOUSE_DATABASE = os.getenv('CLICKHOUSE_DATABASE', 'tictactoe')
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 5000))
    API_DEBUG = os.getenv('API_DEBUG', 'True').lower() == 'true'

# HTTP endpoint for ClickHouse (default port 8123)
CLICKHOUSE_HTTP_PORT = int(os.getenv('CLICKHOUSE_HTTP_PORT', 8123))
CLICKHOUSE_HTTP = os.getenv('CLICKHOUSE_HTTP', f'http://{CLICKHOUSE_HOST}:{CLICKHOUSE_HTTP_PORT}')

# Create requests session for HTTP queries
session = requests.Session()

app = Flask(__name__)

# ----------------------------
# ClickHouse client + table cache
# ----------------------------
try:
    clickhouse_client = Client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD
    )
    try:
        rows = clickhouse_client.execute(f"SHOW TABLES FROM {CLICKHOUSE_DATABASE}")
        TABLE_CACHE = {r[0] for r in rows}
    except Exception as e:
        print(f"Warning: couldn't cache tables: {e}")
        TABLE_CACHE = set()
    print("Connected to ClickHouse")
except Exception as e:
    print(f"Error connecting to ClickHouse: {e}")
    clickhouse_client = None
    TABLE_CACHE = set()

def execute_query(sql: str) -> int:
    """
    Thực thi SQL query và trả về COUNT
    
    Args:
        sql: SQL query string
        
    Returns:
        Số lượng rows (int)
    """
    try:
        response = session.post(
            CLICKHOUSE_HTTP,
            params={
                "user": CLICKHOUSE_USER,
                "password": CLICKHOUSE_PASSWORD,
                "database": CLICKHOUSE_DATABASE
            },
            data=sql,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Query error {response.status_code}: {response.text}")
            return 0
        
        result = response.text.strip()
        if not result:
            return 0
        
        return int(result)
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return 0


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

#==========================================AI Logic==========================================
def best_step(currBoard: list, player: int):
    """
    Tìm nước đi tốt nhất cho AI dựa trên database
    
    Args:
        currBoard: Board hiện tại
        player: Player hiện tại (1 hoặc 2)
        
    Returns:
        Index của nước đi tốt nhất, hoặc -1 nếu không tìm thấy
    """
    start_time = time.time()

    best_move = -1
    win_rate = 0
    lose_rate = 1.0  # Khởi tạo = 1.0 để tìm min
    best_move_by_lose = -1
    
    # Log số ô trống
    empty_cells = sum(1 for cell in currBoard if cell == 0)
    print(f"\nAI đang suy nghĩ... (Còn {empty_cells} ô trống)")
    
    moves_checked = 0
    moves_with_data = 0

    for i in range(len(currBoard)):
        if currBoard[i] != 0:
            continue

        moves_checked += 1
        newBoard = copy.deepcopy(currBoard)
        newBoard[i] = player

        # Convert to canonical form trước khi query
        canonical = canonical_board(newBoard)
        
        # Query với canonical form
        x_win_count = query_odd_table(canonical)
        o_win_count = query_even_table(canonical)
        draw_count = query_draw_table(canonical)
        
        total_count = x_win_count + o_win_count + draw_count
        
        if total_count <= 0:
            continue
        
        moves_with_data += 1
        
        # Tính win rate và lose rate cho player hiện tại
        win_count = x_win_count if player == 1 else o_win_count
        lose_count = o_win_count if player == 1 else x_win_count
        
        current_win_rate = win_count / total_count
        current_lose_rate = lose_count / total_count
        draw_rate = draw_count / total_count
        
        # Log chi tiết
        row = i // 5
        col = i % 5
        print(f"  Ô [{row},{col}] (idx={i}): "
              f"win={current_win_rate:.2%}, lose={current_lose_rate:.2%}, draw={draw_rate:.2%} "
              f"(X:{x_win_count}, O:{o_win_count}, D:{draw_count}, total:{total_count})")
        
        # Tìm nước đi có win_rate cao nhất
        if current_win_rate > win_rate:
            win_rate = current_win_rate
            best_move = i

        # Tìm nước đi có lose_rate thấp nhất
        if current_lose_rate < lose_rate:
            lose_rate = current_lose_rate
            best_move_by_lose = i

    # Nếu không tìm thấy nước thắng, chọn nước ít thua nhất
    if best_move == -1:
        best_move = best_move_by_lose

    elapsed_time = time.time() - start_time
    
    if best_move != -1:
        print(f"\nAI chọn ô {best_move} (row={best_move//5}, col={best_move%5})")
        print(f"   Win rate: {win_rate:.2%}, Lose rate: {lose_rate:.2%}")
    else:
        print(f"\nKhông tìm thấy nước đi tốt trong database")
        # Fallback: chọn ô trống đầu tiên
        for i in range(len(currBoard)):
            if currBoard[i] == 0:
                best_move = i
                break
        if best_move != -1:
            print(f"   Chọn random: ô {best_move} (row={best_move//5}, col={best_move%5})")
    
    print(f"Thời gian suy nghĩ: {elapsed_time:.3f}s")
    print(f"Đã kiểm tra {moves_checked} nước đi, {moves_with_data} có data")
    
    return best_move

@app.route('/api/move', methods=['GET'])
def get_move():
    try:
        board_size = int(request.args.get('boardSize') or request.args.get('size') or '5')
        win_length = int(request.args.get('winLength') or request.args.get('win') or '5')
        next_move = (request.args.get('nextMove') or request.args.get('player') or 'X').upper()
        matrix = request.args.get('matrix') or request.args.get('board')

        if board_size < 3 or board_size > 100:
            return jsonify({'error': 'Board size must be between 3 and 100'}), 400
        if win_length < 3 or win_length > board_size:
            return jsonify({'error': f'Win length must be between 3 and {board_size}'}), 400
        if next_move not in ['X', 'O']:
            return jsonify({'error': 'Next move must be X or O'}), 400

        if matrix:
            try:
                board = parse_board_matrix(matrix, board_size)
            except Exception as e:
                return jsonify({'error': f'Invalid matrix format: {e}'}), 400
        else:
            board = [['-' for _ in range(board_size)] for _ in range(board_size)]

        temperature = float(request.args.get('temperature', '0.1'))

        # Validate board size (current implementation only supports 5x5)
        if board_size != 5:
            return jsonify({'error': 'Board size must be 5 for current implementation'}), 400

        if not clickhouse_client:
            return jsonify({'error': 'Database connection not available'}), 503

        # Convert 2D board (List[List[str]]) to 1D list of ints
        # '-' -> 0, 'X' -> 1, 'O' -> 2
        curr_board_1d = []
        for row in board:
            for cell in row:
                if cell == '-' or cell == '':
                    curr_board_1d.append(0)
                elif cell.upper() == 'X':
                    curr_board_1d.append(1)
                elif cell.upper() == 'O':
                    curr_board_1d.append(2)
                else:
                    curr_board_1d.append(0)
        
        # Convert player string to int (1 for X, 2 for O)
        player_int = 1 if next_move == 'X' else 2
        
        # Call best_step to get the best move index
        best_move_idx = best_step(curr_board_1d, player_int)
        
        if best_move_idx == -1:
            return jsonify({'error': 'Could not determine best move.'}), 404
        
        # Convert index back to row/col
        best_row = best_move_idx // board_size
        best_col = best_move_idx % board_size
        
        # Query statistics for the best move
        test_board = copy.deepcopy(curr_board_1d)
        test_board[best_move_idx] = player_int
        canonical = canonical_board(test_board)
        
        x_win_count = query_odd_table(canonical)
        o_win_count = query_even_table(canonical)
        draw_count = query_draw_table(canonical)
        
        compatible_boards_count = x_win_count + o_win_count + draw_count
        
        # Calculate score and outcome
        score = 0.0
        outcome = 'unknown'
        
        if compatible_boards_count > 0:
            win_count = x_win_count if player_int == 1 else o_win_count
            lose_count = o_win_count if player_int == 1 else x_win_count
            
            win_rate = win_count / compatible_boards_count
            lose_rate = lose_count / compatible_boards_count
            
            score = win_rate - lose_rate
            
            if score > 0.3:
                outcome = 'win'
            elif score < -0.3:
                outcome = 'lose'
            else:
                outcome = 'draw'

        return jsonify({
            'row': best_row,
            'col': best_col,
            'move': next_move,
            'boardSize': board_size,
            'winLength': win_length,
            'outcome': outcome,
            'score': score
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e) or 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    db_status = 'connected' if clickhouse_client else 'disconnected'
    return jsonify({'status': 'ok', 'database': db_status})


if __name__ == '__main__':
    print(f"Starting server on {API_HOST}:{API_PORT} (ClickHouse: {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT})")
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
