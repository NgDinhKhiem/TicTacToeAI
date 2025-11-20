"""
Simple test script for the Tic-Tac-Toe API
"""

import requests
import json

API_URL = "http://localhost:5000"

def test_health():
    """Test health check endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_move_empty_board():
    """Test move on empty board"""
    print("Testing /api/move with empty 5x5 board...")
    params = {
        'boardSize': 5,
        'nextMove': 'X',
        'matrix': '-' * 25  # Empty 5x5 board
    }
    response = requests.get(f"{API_URL}/api/move", params=params)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_move_partial_board():
    """Test move on partial board"""
    print("Testing /api/move with partial board...")
    # Create a board with a few moves
    board = [
        ['X', '-', '-', '-', '-'],
        ['-', 'O', '-', '-', '-'],
        ['-', '-', 'X', '-', '-'],
        ['-', '-', '-', '-', '-'],
        ['-', '-', '-', '-', '-']
    ]
    board_str = ''.join(''.join(row) for row in board)
    
    params = {
        'boardSize': 5,
        'nextMove': 'O',
        'matrix': board_str
    }
    response = requests.get(f"{API_URL}/api/move", params=params)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_move_json_format():
    """Test move with JSON matrix format"""
    print("Testing /api/move with JSON matrix format...")
    board = [
        ['X', '-', '-'],
        ['-', 'O', '-'],
        ['-', '-', 'X']
    ]
    params = {
        'boardSize': 3,
        'nextMove': 'O',
        'matrix': json.dumps(board)
    }
    response = requests.get(f"{API_URL}/api/move", params=params)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

if __name__ == '__main__':
    print("=" * 60)
    print("Tic-Tac-Toe API Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_move_empty_board()
        test_move_partial_board()
        test_move_json_format()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API server.")
        print("Make sure the server is running: python api_server.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

