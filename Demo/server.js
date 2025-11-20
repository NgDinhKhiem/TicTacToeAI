const express = require('express');
const cors = require('cors');
const http = require('http');
const { URL } = require('url');
const app = express();
const PORT = process.env.PORT || 3000;
const API_SERVER_URL = process.env.API_SERVER_URL || 'http://localhost:5000';

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Helper function to make HTTP requests
function makeRequest(url, options = {}) {
    return new Promise((resolve, reject) => {
        const urlObj = new URL(url);
        const port = urlObj.port || (urlObj.protocol === 'https:' ? 443 : 80);
        
        const requestOptions = {
            hostname: urlObj.hostname,
            port: port,
            path: urlObj.pathname + urlObj.search,
            method: options.method || 'GET',
            headers: {
                'User-Agent': 'Node.js Proxy',
                ...options.headers
            },
            timeout: 10000 // 10 second timeout
        };

        const req = http.request(requestOptions, (res) => {
            let data = '';
            res.on('data', (chunk) => { data += chunk; });
            res.on('end', () => {
                try {
                    const jsonData = JSON.parse(data);
                    resolve({ status: res.statusCode, data: jsonData });
                } catch (e) {
                    resolve({ status: res.statusCode, data: data });
                }
            });
        });

        req.on('error', (error) => {
            reject(error);
        });

        req.on('timeout', () => {
            req.destroy();
            reject(new Error('Request timeout'));
        });

        if (options.body) {
            req.write(options.body);
        }
        req.end();
    });
}

// Proxy endpoint to forward API requests to Python Flask server
app.get('/api/move', async (req, res) => {
    try {
        const queryString = new URLSearchParams(req.query).toString();
        const apiUrl = `${API_SERVER_URL}/api/move?${queryString}`;
        
        console.log(`[${new Date().toISOString()}] Proxying request to: ${apiUrl}`);
        console.log(`  Query params:`, req.query);
        
        const response = await makeRequest(apiUrl);
        
        console.log(`  Response status: ${response.status}`);
        res.status(response.status).json(response.data);
    } catch (error) {
        console.error(`[${new Date().toISOString()}] Error proxying request to API server:`, error);
        console.error(`  Error code: ${error.code}`);
        console.error(`  Error message: ${error.message}`);
        
        // Provide more helpful error messages
        let errorMessage = 'Failed to connect to API server';
        if (error.code === 'ECONNREFUSED') {
            errorMessage = `Cannot connect to API server at ${API_SERVER_URL}. Make sure the Python Flask server is running on port 5000.`;
        } else if (error.code === 'ETIMEDOUT') {
            errorMessage = 'API server request timed out';
        }
        
        res.status(500).json({ 
            error: errorMessage,
            message: error.message,
            code: error.code,
            apiServerUrl: API_SERVER_URL
        });
    }
});

// Proxy health check endpoint
app.get('/api/health', async (req, res) => {
    try {
        const apiUrl = `${API_SERVER_URL}/health`;
        const response = await makeRequest(apiUrl);
        res.status(response.status).json(response.data);
    } catch (error) {
        res.status(500).json({ 
            error: 'Failed to connect to API server',
            message: error.message 
        });
    }
});

// Constants
const EMPTY = '-';
const X = 'X';
const O = 'O';

// Check if a line has a winning sequence
function checkLine(line, winLength) {
    let count = 1;
    let prev = null;
    for (const token of line) {
        if (token === EMPTY) {
            prev = null;
            count = 0;
            continue;
        }
        if (token === prev) {
            count++;
            if (count >= winLength) return true;
        } else {
            prev = token;
            count = 1;
        }
    }
    return false;
}

// Check if the game has ended (win or draw)
function isGameEnded(board, winLength) {
    const n = board.length;
    
    // Check rows
    for (let i = 0; i < n; i++) {
        if (checkLine(board[i], winLength)) return true;
    }
    
    // Check columns
    for (let i = 0; i < n; i++) {
        const col = board.map(row => row[i]);
        if (checkLine(col, winLength)) return true;
    }
    
    // Check diagonals (top-left to bottom-right)
    for (let start = 0; start <= 2 * (n - 1); start++) {
        const diag = [];
        for (let i = 0; i < n; i++) {
            const j = start - i;
            if (j >= 0 && j < n) {
                diag.push(board[i][j]);
            }
        }
        if (diag.length >= winLength && checkLine(diag, winLength)) return true;
    }
    
    // Check diagonals (top-right to bottom-left)
    for (let start = -(n - 1); start <= (n - 1); start++) {
        const diag = [];
        for (let i = 0; i < n; i++) {
            const j = i - start;
            if (j >= 0 && j < n) {
                diag.push(board[i][j]);
            }
        }
        if (diag.length >= winLength && checkLine(diag, winLength)) return true;
    }
    
    return false;
}

// Check if board is full
function isBoardFull(board) {
    return board.every(row => row.every(cell => cell !== EMPTY));
}

// Count tokens on board
function countTokens(board) {
    let xCount = 0, oCount = 0;
    for (const row of board) {
        for (const cell of row) {
            if (cell === X) xCount++;
            else if (cell === O) oCount++;
        }
    }
    return { xCount, oCount };
}

// Minimax algorithm with alpha-beta pruning
function minimax(board, winLength, depth, isMaximizing, alpha, beta, player) {
    const opponent = player === X ? O : X;
    
    if (isGameEnded(board, winLength)) {
        // Check who won
        const { xCount, oCount } = countTokens(board);
        const lastPlayer = xCount > oCount ? X : O;
        if (lastPlayer === player) return 1000 - depth; // Win for AI
        return depth - 1000; // Loss for AI
    }
    
    if (isBoardFull(board)) {
        return 0; // Draw
    }
    
    if (isMaximizing) {
        let maxEval = -Infinity;
        for (let i = 0; i < board.length; i++) {
            for (let j = 0; j < board.length; j++) {
                if (board[i][j] === EMPTY) {
                    board[i][j] = player;
                    const eval = minimax(board, winLength, depth + 1, false, alpha, beta, player);
                    board[i][j] = EMPTY;
                    maxEval = Math.max(maxEval, eval);
                    alpha = Math.max(alpha, eval);
                    if (beta <= alpha) break; // Alpha-beta pruning
                }
            }
            if (beta <= alpha) break;
        }
        return maxEval;
    } else {
        let minEval = Infinity;
        for (let i = 0; i < board.length; i++) {
            for (let j = 0; j < board.length; j++) {
                if (board[i][j] === EMPTY) {
                    board[i][j] = opponent;
                    const eval = minimax(board, winLength, depth + 1, true, alpha, beta, player);
                    board[i][j] = EMPTY;
                    minEval = Math.min(minEval, eval);
                    beta = Math.min(beta, eval);
                    if (beta <= alpha) break; // Alpha-beta pruning
                }
            }
            if (beta <= alpha) break;
        }
        return minEval;
    }
}

// Get the best move using minimax
function getBestMove(board, winLength, player) {
    let bestMove = null;
    let bestValue = -Infinity;
    
    // Limit depth for large boards to avoid timeout
    const maxDepth = board.length <= 5 ? 10 : (board.length <= 10 ? 5 : 3);
    
    for (let i = 0; i < board.length; i++) {
        for (let j = 0; j < board.length; j++) {
            if (board[i][j] === EMPTY) {
                board[i][j] = player;
                const moveValue = minimax(board, winLength, 0, false, -Infinity, Infinity, player);
                board[i][j] = EMPTY;
                
                if (moveValue > bestValue) {
                    bestValue = moveValue;
                    bestMove = { row: i, col: j };
                }
            }
        }
    }
    
    return bestMove;
}

// // API endpoint: GET /api/move
// app.get('/api/move', (req, res) => {
//     try {
//         const boardSize = parseInt(req.query.boardSize || req.query.size || '3');
//         const winLength = parseInt(req.query.winLength || req.query.win || '3');
//         const nextMove = (req.query.nextMove || req.query.player || 'X').toUpperCase();
//         const matrix = req.query.matrix || req.query.board;
        
//         // Validate inputs
//         if (boardSize < 3 || boardSize > 100) {
//             return res.status(400).json({ error: 'Board size must be between 3 and 100' });
//         }
        
//         if (winLength < 3 || winLength > boardSize) {
//             return res.status(400).json({ error: `Win length must be between 3 and ${boardSize}` });
//         }
        
//         if (nextMove !== X && nextMove !== O) {
//             return res.status(400).json({ error: 'Next move must be X or O' });
//         }
        
//         // Parse matrix
//         let board;
//         if (matrix) {
//             try {
//                 // Try parsing as JSON first
//                 board = JSON.parse(matrix);
//             } catch (e) {
//                 // If not JSON, try parsing as string representation
//                 // Format: "X-O--X-O" or "X,O,-,X,O,-"
//                 const cells = matrix.split(/[,\s-]+/).filter(c => c);
//                 if (cells.length === boardSize * boardSize) {
//                     board = [];
//                     for (let i = 0; i < boardSize; i++) {
//                         board.push(cells.slice(i * boardSize, (i + 1) * boardSize));
//                     }
//                 } else {
//                     throw new Error('Invalid matrix format');
//                 }
//             }
//         } else {
//             // Create empty board
//             board = Array(boardSize).fill(null).map(() => Array(boardSize).fill(EMPTY));
//         }
        
//         // Validate board dimensions
//         if (!Array.isArray(board) || board.length !== boardSize) {
//             return res.status(400).json({ error: 'Board dimensions do not match board size' });
//         }
        
//         for (let i = 0; i < board.length; i++) {
//             if (!Array.isArray(board[i]) || board[i].length !== boardSize) {
//                 return res.status(400).json({ error: 'Board dimensions do not match board size' });
//             }
//         }
        
//         // Normalize board (convert empty cells to EMPTY constant)
//         board = board.map(row => row.map(cell => {
//             if (cell === null || cell === undefined || cell === '' || cell === ' ') {
//                 return EMPTY;
//             }
//             return cell.toUpperCase();
//         }));
        
//         // Check if game is already ended
//         if (isGameEnded(board, winLength)) {
//             return res.status(400).json({ error: 'Game has already ended' });
//         }
        
//         if (isBoardFull(board)) {
//             return res.status(400).json({ error: 'Board is full' });
//         }
        
//         // Get best move
//         const bestMove = getBestMove(board, winLength, nextMove);
        
//         if (!bestMove) {
//             return res.status(400).json({ error: 'No valid moves available' });
//         }
        
//         res.json({
//             row: bestMove.row,
//             col: bestMove.col,
//             move: nextMove,
//             boardSize: boardSize,
//             winLength: winLength
//         });
        
//     } catch (error) {
//         console.error('Error processing move:', error);
//         res.status(500).json({ error: error.message || 'Internal server error' });
//     }
// });

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

// Status endpoint to check API server connectivity
app.get('/api/status', async (req, res) => {
    try {
        const response = await makeRequest(`${API_SERVER_URL}/health`);
        res.json({
            status: 'ok',
            apiServer: {
                url: API_SERVER_URL,
                reachable: true,
                status: response.status,
                response: response.data
            }
        });
    } catch (error) {
        res.status(503).json({
            status: 'error',
            apiServer: {
                url: API_SERVER_URL,
                reachable: false,
                error: error.message,
                code: error.code
            },
            message: 'API server is not reachable. Make sure the Python Flask server is running on port 5000.'
        });
    }
});

// Test API server connectivity on startup
async function testAPIConnection() {
    try {
        const response = await makeRequest(`${API_SERVER_URL}/health`);
        console.log(`✓ API server is reachable at ${API_SERVER_URL}`);
        console.log(`  Status: ${response.status}, Response:`, response.data);
    } catch (error) {
        console.warn(`⚠ Warning: Cannot reach API server at ${API_SERVER_URL}`);
        console.warn(`  Error: ${error.message}`);
        console.warn(`  Make sure the Python API server is running on port 5000`);
    }
}

app.listen(PORT, async () => {
    console.log(`Tic-Tac-Toe AI Server running on http://localhost:${PORT}`);
    console.log(`API Server URL: ${API_SERVER_URL}`);
    await testAPIConnection();
});

