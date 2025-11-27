"""
Configuration file for ML module
"""

# ClickHouse connection settings
CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_PORT = 9000
CLICKHOUSE_USER = 'admin'
CLICKHOUSE_PASSWORD = 'Helloworld'
CLICKHOUSE_DATABASE = 'tictactoe'

# Model configuration
# Note: BOARD_SIZE will be auto-detected from data during training
# This is just a default/target size (for API, use actual data size)
BOARD_SIZE = 10  # Default/target board size (actual size detected from data)
WIN_LENGTH = 5   # 5 in a row to win

# Training configuration
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EMBEDDING_DIM = 128
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'

# Evaluation configuration
EVAL_INTERVAL = 5  # Evaluate every N epochs
EVAL_SPLIT = 0.1  # Use 10% of data for evaluation
LOG_INTERVAL = 50  # Log metrics every N batches

# Model architecture choices
MODEL_TYPE = 'CNN'  # 'CNN' or 'MLP'
USE_AUTOENCODER = True  # Use autoencoder for unsupervised learning

# Paths
MODEL_SAVE_DIR = 'models'
CHECKPOINT_DIR = 'checkpoints'

# Weights & Biases (wandb) configuration
WANDB_PROJECT = 'tictactoe_ml'
WANDB_GROUP = 'unsupervised_learning'
WANDB_MODE = 'online'  # 'online', 'offline', or 'disabled'
WANDB_NAME = None  # None for auto-generated name

