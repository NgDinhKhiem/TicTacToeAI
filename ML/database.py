"""
ClickHouse database connector for loading board data
"""

from clickhouse_driver import Client
from typing import List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ClickHouseConnector:
    """Connector for ClickHouse database to fetch board states"""
    
    def __init__(self, host='localhost', port=9000, user='admin', 
                 password='Helloworld', database='tictactoe'):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to ClickHouse"""
        try:
            self.client = Client(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            logger.info(f"Connected to ClickHouse at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Error connecting to ClickHouse: {e}")
            raise
    
    def get_table_names(self, board_size: int = 5) -> List[str]:
        """Get all table names for a given board size"""
        query = f"SHOW TABLES FROM {self.database} LIKE 'ttt_{board_size}_%'"
        try:
            result = self.client.execute(query)
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Error fetching table names: {e}")
            return []
    
    def get_boards_from_table(self, table_name: str, limit: Optional[int] = None, 
                              offset: int = 0) -> List[Tuple[np.ndarray, str]]:
        """
        Fetch boards from a specific table
        
        Args:
            table_name: Name of the table
            limit: Maximum number of boards to fetch
            offset: Offset for pagination
            
        Returns:
            List of tuples (board_array, win_actor) where:
            - board_array: numpy array of shape (board_size, board_size)
            - win_actor: 'X', 'O', or 'D' (draw)
        """
        # Build column list for board positions
        # For 5x5: i11, i12, ..., i55
        # For 10x10: i11, i12, ..., i10_10
        board_size = self._infer_board_size_from_table(table_name)
        columns = []
        for i in range(1, board_size + 1):
            for j in range(1, board_size + 1):
                columns.append(f'i{i}{j}')
        
        columns_str = ', '.join(columns)
        query = f"""
            SELECT {columns_str}, win_actor 
            FROM {self.database}.{table_name}
        """
        
        if limit:
            query += f" LIMIT {limit} OFFSET {offset}"
        
        try:
            result = self.client.execute(query)
            boards = []
            
            for row in result:
                board_values = row[:-1]  # All columns except win_actor
                win_actor = row[-1] if row[-1] else 'D'
                
                # Convert to numpy array
                board_array = np.array([
                    [self._char_to_int(board_values[i * board_size + j]) 
                     for j in range(board_size)]
                    for i in range(board_size)
                ])
                
                boards.append((board_array, win_actor))
            
            return boards
        except Exception as e:
            logger.error(f"Error fetching boards from {table_name}: {e}")
            return []
    
    def get_all_boards(self, board_size: int = 5, max_boards_per_table: Optional[int] = None) -> List[Tuple[np.ndarray, str]]:
        """
        Get all boards from all tables for a given board size
        
        Args:
            board_size: Size of the board (5 for 5x5, etc.)
            max_boards_per_table: Maximum boards to fetch per table
            
        Returns:
            List of (board_array, win_actor) tuples
        """
        table_names = self.get_table_names(board_size)
        all_boards = []
        
        for table_name in table_names:
            logger.info(f"Loading boards from {table_name}...")
            boards = self.get_boards_from_table(table_name, limit=max_boards_per_table)
            all_boards.extend(boards)
            logger.info(f"Loaded {len(boards)} boards from {table_name}")
        
        logger.info(f"Total boards loaded: {len(all_boards)}")
        return all_boards
    
    def _infer_board_size_from_table(self, table_name: str) -> int:
        """Infer board size from table name"""
        # Table names like ttt_5_l9, ttt_5_draw
        # Extract the number after ttt_
        try:
            parts = table_name.split('_')
            if len(parts) >= 2:
                return int(parts[1])
        except:
            pass
        return 5  # Default
    
    def _char_to_int(self, char: str) -> int:
        """Convert board character to integer encoding"""
        char = char.strip().upper()
        if char == 'X':
            return 1
        elif char == 'O':
            return 2
        else:  # '-' or empty
            return 0
    
    def close(self):
        """Close the connection"""
        if self.client:
            self.client.disconnect()

