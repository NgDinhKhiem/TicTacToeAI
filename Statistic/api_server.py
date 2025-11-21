from flask import Flask, request, jsonify
from clickhouse_driver import Client
import copy
import os
import time
import math
import json
import requests
from typing import List, Tuple, Optional, Dict, Callable
import ttt_100;

# Import board utilities module
from board_utils import (
    parse_board_matrix,
    query_odd_table,
    query_even_table,
    query_draw_table,
    canonical_board,
    set_execute_query
)

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
    API_PORT = int(os.getenv('API_PORT', 5050))
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

set_execute_query(execute_query)
ttt_100.set_execute_query(execute_query)


#==========================================AI Logic==========================================
def best_step(currBoard: list, player: int):
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
        last_move_row_str = request.args.get('last_move_row')
        last_move_col_str = request.args.get('last_move_col')
        
        # Convert to integers if provided, otherwise None
        last_move_row = int(last_move_row_str) if last_move_row_str is not None else None
        last_move_col = int(last_move_col_str) if last_move_col_str is not None else None
        
        print(f"Row: {last_move_row}, Col: {last_move_col}")

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

        if not clickhouse_client:
            return jsonify({'error': 'Database connection not available'}), 503

        # Validate board size (current implementation only supports 5x5)
        if board_size > 5:
            curr_board_int_2d: list[list[int]] = []
            for row in board:
                row_int = []
                for cell in row:
                    if cell == '-' or cell == '':
                        row_int.append(0)
                    elif cell.upper() == 'X':
                        row_int.append(1)
                    elif cell.upper() == 'O':
                        row_int.append(2)
                    else:
                        row_int.append(0)
                curr_board_int_2d.append(row_int)

            # Handle None values for last_move_row and last_move_col
            # If not provided, find the last move from the board state
            if last_move_row is None or last_move_col is None:
                # Find any non-empty cell on the board to use as reference
                found_move = False
                for r in range(board_size):
                    for c in range(board_size):
                        if curr_board_int_2d[r][c] != 0:
                            if last_move_row is None:
                                last_move_row = r
                            if last_move_col is None:
                                last_move_col = c
                            found_move = True
                            break
                    if found_move:
                        break
                
                # If board is empty, use center as default
                if not found_move:
                    if last_move_row is None:
                        last_move_row = board_size // 2
                    if last_move_col is None:
                        last_move_col = board_size // 2
                    print(f"No moves found on board, using center: row={last_move_row}, col={last_move_col}")
                else:
                    print(f"Found last move from board state: row={last_move_row}, col={last_move_col}")
            
            # Validate last_move coordinates are within bounds
            if last_move_row < 0 or last_move_row >= board_size:
                print(f"Warning: last_move_row {last_move_row} out of bounds, clamping to valid range")
                last_move_row = max(0, min(board_size - 1, last_move_row))
            if last_move_col < 0 or last_move_col >= board_size:
                print(f"Warning: last_move_col {last_move_col} out of bounds, clamping to valid range")
                last_move_col = max(0, min(board_size - 1, last_move_col))
            
            # Convert player string to int (1 for X, 2 for O)
            player_int = 1 if next_move == 'X' else 2
            
            print(f"Calling best_steps_unlimited with player={player_int}, last_move_col={last_move_col}, last_move_row={last_move_row}")
            try:
                (r, c) = ttt_100.best_steps_unlimited(curr_board_int_2d, player_int, last_move_col, last_move_row)
                
                # Handle case where function returns -1 (no move found)
                if r == -1 or c == -1:
                    print("best_steps_unlimited returned -1, finding first empty cell as fallback")
                    # Find first empty cell as fallback
                    for r_fallback in range(board_size):
                        for c_fallback in range(board_size):
                            if curr_board_int_2d[r_fallback][c_fallback] == 0:
                                return jsonify({
                                    'row': r_fallback,
                                    'col': c_fallback,
                                })
                    # If board is full, return center
                    return jsonify({
                        'row': board_size // 2,
                        'col': board_size // 2,
                    })
                
                return jsonify({
                    'row': r,
                    'col': c,
                })
            except Exception as e:
                import traceback
                print(f"Error in best_steps_unlimited: {e}")
                traceback.print_exc()
                # Fallback: find first empty cell
                for r_fallback in range(board_size):
                    for c_fallback in range(board_size):
                        if curr_board_int_2d[r_fallback][c_fallback] == 0:
                            return jsonify({
                                'row': r_fallback,
                                'col': c_fallback,
                            })
                # If board is full, return center
                return jsonify({
                    'row': board_size // 2,
                    'col': board_size // 2,
                })

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
        
        best_row = best_move_idx // board_size
        best_col = best_move_idx % board_size
        
        return jsonify({
            'row': best_row,
            'col': best_col,
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
