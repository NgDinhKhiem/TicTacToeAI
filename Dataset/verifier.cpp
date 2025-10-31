#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <iomanip>

const char EMPTY_TOKEN = '-';

// Check if a line has a winning sequence
bool checkLine(const std::vector<char>& line, int winLen) {
    int count = 1;
    char prev = '\0';
    
    for (char token : line) {
        if (token == EMPTY_TOKEN) {
            prev = '\0';
            count = 0;
            continue;
        }
        if (token == prev) {
            count++;
            if (count >= winLen) {
                return true;
            }
        } else {
            prev = token;
            count = 1;
        }
    }
    return false;
}

// Check if the game has ended (win or draw)
bool isBoardEnded(const std::vector<std::vector<char>>& board) {
    int n = board.size();
    int winLen = (n == 3) ? 3 : 5;
    
    // Check rows
    for (int i = 0; i < n; i++) {
        if (checkLine(board[i], winLen)) {
            return true;
        }
    }
    
    // Check columns
    for (int i = 0; i < n; i++) {
        std::vector<char> column;
        for (int j = 0; j < n; j++) {
            column.push_back(board[j][i]);
        }
        if (checkLine(column, winLen)) {
            return true;
        }
    }
    
    // Check diagonals (top-left to bottom-right direction)
    for (int p = 0; p < n * 2 - 1; p++) {
        std::vector<char> diag1;
        int start = std::max(0, p - n + 1);
        int end = std::min(p + 1, n);
        for (int i = start; i < end; i++) {
            int j = p - i;
            if (j >= 0 && j < n) {
                diag1.push_back(board[i][j]);
            }
        }
        if (diag1.size() >= static_cast<size_t>(winLen) && checkLine(diag1, winLen)) {
            return true;
        }
    }
    
    // Check diagonals (top-right to bottom-left direction)
    for (int p = -n + 1; p < n; p++) {
        std::vector<char> diag2;
        int start = std::max(0, p);
        int end = std::min(n, n + p);
        for (int i = start; i < end; i++) {
            int j = i - p;
            if (j >= 0 && j < n) {
                diag2.push_back(board[i][j]);
            }
        }
        if (diag2.size() >= static_cast<size_t>(winLen) && checkLine(diag2, winLen)) {
            return true;
        }
    }
    
    return false;
}

// Convert hash string to 2D board
std::vector<std::vector<char>> hashToBoard(const std::string& hashStr, int boardSize) {
    std::vector<std::vector<char>> board(boardSize, std::vector<char>(boardSize));
    for (int i = 0; i < boardSize; i++) {
        for (int j = 0; j < boardSize; j++) {
            board[i][j] = hashStr[i * boardSize + j];
        }
    }
    return board;
}

// Load unique boards from file
std::vector<std::vector<std::vector<char>>> loadUniqueBoards(const std::string& filepath, int boardSize) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filepath << std::endl;
        exit(1);
    }
    
    std::unordered_set<std::string> uniqueHashes;
    std::string line;
    int totalLines = 0;
    
    while (std::getline(file, line)) {
        // Trim whitespace
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (!line.empty()) {
            uniqueHashes.insert(line);
            totalLines++;
        }
    }
    file.close();
    
    std::cout << "Loaded " << totalLines << " lines, " << uniqueHashes.size() << " unique boards." << std::endl;
    
    std::vector<std::vector<std::vector<char>>> boards;
    for (const std::string& hashStr : uniqueHashes) {
        if (hashStr.length() != static_cast<size_t>(boardSize * boardSize)) {
            std::cerr << "Error: Invalid board length for " << hashStr << std::endl;
            continue;
        }
        boards.push_back(hashToBoard(hashStr, boardSize));
    }
    
    return boards;
}

// Count winning boards with progress
int countWinningBoards(const std::string& filepath, int boardSize) {
    auto boards = loadUniqueBoards(filepath, boardSize);
    int winCount = 0;
    
    std::cout << "Checking boards for wins..." << std::endl;
    
    size_t total = boards.size();
    size_t progressStep = total / 100;  // Update progress every 1%
    if (progressStep == 0) progressStep = 1;
    
    for (size_t i = 0; i < boards.size(); i++) {
        if (isBoardEnded(boards[i])) {
            winCount++;
        }
        
        // Show progress
        if (i % progressStep == 0 || i == boards.size() - 1) {
            double percentage = (double)(i + 1) / total * 100.0;
            std::cout << "\rProcessing boards: " << (i + 1) << "/" << total 
                      << " (" << std::fixed << std::setprecision(1) << percentage << "%)" << std::flush;
        }
    }
    
    std::cout << std::endl << std::endl;
    std::cout << "Total unique boards: " << boards.size() << std::endl;
    std::cout << "Winning boards: " << winCount << std::endl;
    std::cout << "Non-winning boards: " << (boards.size() - winCount) << std::endl;
    
    return winCount;
}

int main(int argc, char* argv[]) {
    std::string filepath = "unique.out";
    int boardSize = 5;
    
    // Parse command line arguments
    if (argc > 1) {
        filepath = argv[1];
    }
    if (argc > 2) {
        boardSize = std::stoi(argv[2]);
    }
    
    std::cout << "Processing file: " << filepath << std::endl;
    std::cout << "Board size: " << boardSize << "x" << boardSize << std::endl << std::endl;
    
    countWinningBoards(filepath, boardSize);
    
    return 0;
}

