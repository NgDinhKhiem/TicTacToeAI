const express = require('express');
const cors = require('cors');
const http = require('http');
const { URL } = require('url');
const app = express();
const PORT = process.env.PORT || 3030;
const API_SERVER_URL = process.env.API_SERVER_URL || 'http://localhost:5050';

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Helper function to pretty print matrix
function prettyPrintMatrix(matrix) {
    if (!matrix || !Array.isArray(matrix)) {
        return 'Invalid matrix';
    }
    
    const rows = matrix.length;
    if (rows === 0) {
        return 'Empty matrix';
    }
    
    const cols = matrix[0]?.length || 0;
    let output = `\n  Board (${rows}x${cols}):\n`;
    
    // Add column headers
    output += '    ';
    for (let c = 0; c < cols; c++) {
        output += `${c.toString().padStart(3)} `;
    }
    output += '\n';
    
    // Add rows with row numbers
    for (let r = 0; r < rows; r++) {
        output += `  ${r.toString().padStart(2)} `;
        for (let c = 0; c < cols; c++) {
            const cell = matrix[r]?.[c] || '-';
            // Replace '-' with '.' for better visibility, keep X and O as is
            const display = cell === '-' ? '.' : cell;
            output += `${display.toString().padStart(3)} `;
        }
        output += '\n';
    }
    
    return output;
}

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
            timeout: 100000000 // 10 second timeout
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
        
        // Parse and pretty print matrix if present
        if (req.query.matrix) {
            try {
                const matrix = JSON.parse(decodeURIComponent(req.query.matrix));
                console.log(prettyPrintMatrix(matrix));
            } catch (e) {
                console.log(`  Matrix (raw): ${req.query.matrix}`);
            }
        }
        
        // Log other query params (excluding matrix to avoid clutter)
        const otherParams = { ...req.query };
        delete otherParams.matrix;
        if (Object.keys(otherParams).length > 0) {
            console.log(`  Other params:`, otherParams);
        }
        
        // Log last move parameters specifically
        if (req.query.last_move_row !== undefined || req.query.last_move_col !== undefined) {
            console.log(`  Last move: row=${req.query.last_move_row}, col=${req.query.last_move_col}`);
        } else {
            console.log(`  No last move parameters provided`);
        }
        
        const response = await makeRequest(apiUrl);
        
        console.log(`  Response status: ${response.status}`);
        if (response.data && response.data.row !== undefined && response.data.col !== undefined) {
            console.log(`  AI move: row=${response.data.row}, col=${response.data.col}`);
        }
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
        console.warn(`  Make sure the Python API server is running on port 5050`);
    }
}

app.listen(PORT, async () => {
    console.log(`Tic-Tac-Toe AI Server running on http://localhost:${PORT}`);
    console.log(`API Server URL: ${API_SERVER_URL}`);
    await testAPIConnection();
});

