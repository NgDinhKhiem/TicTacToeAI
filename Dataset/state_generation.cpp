#include <iostream>
#include <vector>
#include <string>
#include <unordered_set>
#include <queue>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <memory>

#include <map>

struct Config {
    int board_size = 3;
    int n_workers = 7;
    size_t chunk_size = 2000000;
    size_t buffer_size = 5000;
    int batch_size_divisor = 2;
    int min_batch_size = 50;
    std::string output_file = "data/unique_cpp.txt";
    char empty_token = '-';
    std::vector<char> player_tokens = {'X', 'O'};
};

// Global configuration
Config config;

// Board type definitions
using Board = std::vector<std::string>;
using BoardStr = std::string;

// Statistics
struct Statistics {
    std::atomic<size_t> wins{0};
    std::atomic<size_t> draws{0};
    std::atomic<size_t> ongoing{0};
    std::atomic<size_t> unique_boards{0};
};

Statistics global_stats;
std::mutex cout_mutex;
std::mutex file_mutex;
std::mutex seen_mutex;

// Simple JSON parser
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\"");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r\",");
    return str.substr(first, (last - first + 1));
}

void load_config(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Config file not found, using defaults." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        
        std::string key = trim(line.substr(0, colon));
        std::string value = trim(line.substr(colon + 1));
        
        if (key == "board_size") config.board_size = std::stoi(value);
        else if (key == "n_workers") config.n_workers = std::stoi(value);
        else if (key == "chunk_size") config.chunk_size = std::stoull(value);
        else if (key == "buffer_size") config.buffer_size = std::stoull(value);
        else if (key == "batch_size_divisor") config.batch_size_divisor = std::stoi(value);
        else if (key == "min_batch_size") config.min_batch_size = std::stoi(value);
        else if (key == "output_file") config.output_file = value;
        else if (key == "empty_token") {
            if (!value.empty()) config.empty_token = value[0];
        }
    }
}

// Board utilities
BoardStr hash_board(const Board& board) {
    std::string result;
    for (const auto& row : board) {
        result += row;
    }
    return result;
}

Board string_to_board(const BoardStr& str, int size) {
    Board board(size);
    for (int i = 0; i < size; i++) {
        board[i] = str.substr(i * size, size);
    }
    return board;
}

Board rotate_90(const Board& board) {
    int n = board.size();
    Board rotated(n, std::string(n, config.empty_token));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            rotated[i][j] = board[n - 1 - j][i];
        }
    }
    return rotated;
}

Board flip_horizontal(const Board& board) {
    Board flipped = board;
    for (auto& row : flipped) {
        std::reverse(row.begin(), row.end());
    }
    return flipped;
}

std::vector<Board> get_all_symmetries(const Board& board) {
    std::vector<Board> variants;
    Board current = board;
    
    // 4 rotations
    for (int i = 0; i < 4; i++) {
        variants.push_back(current);
        current = rotate_90(current);
    }
    
    // 4 rotations of horizontal flip
    Board flipped = flip_horizontal(board);
    current = flipped;
    for (int i = 0; i < 4; i++) {
        variants.push_back(current);
        current = rotate_90(current);
    }
    
    return variants;
}

BoardStr get_canonical_form(const Board& board) {
    auto symmetries = get_all_symmetries(board);
    std::vector<BoardStr> hashes;
    for (const auto& sym : symmetries) {
        hashes.push_back(hash_board(sym));
    }
    return *std::min_element(hashes.begin(), hashes.end());
}

// Game logic
bool check_line(const std::vector<char>& line, int win_len) {
    int count = 0;
    char prev = '\0';
    
    for (char token : line) {
        if (token == config.empty_token) {
            prev = '\0';
            count = 0;
            continue;
        }
        if (token == prev) {
            count++;
            if (count >= win_len) return true;
        } else {
            prev = token;
            count = 1;
        }
    }
    return false;
}

bool is_board_ended(const Board& board) {
    int n = board.size();
    int win_len = std::min(n, 5);
    
    // Check rows
    for (int i = 0; i < n; i++) {
        std::vector<char> row(board[i].begin(), board[i].end());
        if (check_line(row, win_len)) return true;
    }
    
    // Check columns
    for (int i = 0; i < n; i++) {
        std::vector<char> col;
        for (int j = 0; j < n; j++) {
            col.push_back(board[j][i]);
        }
        if (check_line(col, win_len)) return true;
    }
    
    // Check diagonals (top-left to bottom-right)
    for (int p = 0; p < n * 2 - 1; p++) {
        std::vector<char> diag;
        for (int i = std::max(0, p - n + 1); i < std::min(p + 1, n); i++) {
            int j = p - i;
            if (j >= 0 && j < n) {
                diag.push_back(board[i][j]);
            }
        }
        if (diag.size() >= (size_t)win_len && check_line(diag, win_len)) return true;
    }
    
    // Check diagonals (top-right to bottom-left)
    for (int p = -n + 1; p < n; p++) {
        std::vector<char> diag;
        for (int i = std::max(0, p); i < std::min(n, n + p); i++) {
            int j = i - p;
            if (j >= 0 && j < n) {
                diag.push_back(board[i][j]);
            }
        }
        if (diag.size() >= (size_t)win_len && check_line(diag, win_len)) return true;
    }
    
    return false;
}

bool is_board_full(const Board& board) {
    for (const auto& row : board) {
        for (char cell : row) {
            if (cell == config.empty_token) return false;
        }
    }
    return true;
}

std::pair<int, int> count_tokens(const Board& board) {
    int x_count = 0, o_count = 0;
    for (const auto& row : board) {
        for (char cell : row) {
            if (cell == 'X') x_count++;
            else if (cell == 'O') o_count++;
        }
    }
    return {x_count, o_count};
}

// Batch processing result
struct BatchResult {
    size_t wins = 0;
    size_t draws = 0;
    size_t ongoing = 0;
    std::vector<BoardStr> new_children;
};

BatchResult process_batch(const std::vector<Board>& boards) {
    BatchResult result;
    
    for (const auto& board : boards) {
        bool has_winner = is_board_ended(board);
        bool is_full = is_board_full(board);
        bool game_ended = has_winner || is_full;
        
        if (game_ended) {
            if (has_winner) {
                result.wins++;
            } else {
                result.draws++;
            }
            continue;
        }
        
        // Ongoing game - generate children
        result.ongoing++;
        auto [x_count, o_count] = count_tokens(board);
        char current_token = (x_count == o_count) ? 'X' : 'O';
        
        for (size_t i = 0; i < board.size(); i++) {
            for (size_t j = 0; j < board[i].size(); j++) {
                if (board[i][j] == config.empty_token) {
                    Board new_board = board;
                    new_board[i][j] = current_token;
                    
                    BoardStr canonical = get_canonical_form(new_board);
                    result.new_children.push_back(canonical);
                }
            }
        }
    }
    
    return result;
}

// Board state manager
class BoardStateMap {
private:
    std::vector<BoardStr> buffer;
    std::ofstream file;
    int board_size;
    
public:
    BoardStateMap(const std::string& filename, int size) : board_size(size) {
        file.open(filename, std::ios::trunc);
        buffer.reserve(config.buffer_size);
    }
    
    void add_to_buffer(const BoardStr& board_str) {
        std::lock_guard<std::mutex> lock(file_mutex);
        buffer.push_back(board_str);
        if (buffer.size() >= config.buffer_size) {
            flush();
        }
    }
    
    void flush() {
        if (!buffer.empty()) {
            for (const auto& board_str : buffer) {
                // Write board in readable format (rows separated by spaces)
                for (int i = 0; i < board_size; i++) {
                    file << board_str.substr(i * board_size, board_size);
                    if (i < board_size - 1) {
                        file << " ";
                    }
                }
                file << '\n';
            }
            buffer.clear();
        }
    }
    
    void close() {
        std::lock_guard<std::mutex> lock(file_mutex);
        flush();
        file.close();
    }
    
    ~BoardStateMap() {
        if (file.is_open()) {
            flush();
            file.close();
        }
    }
};

// Progress display
void display_progress(size_t processed, size_t queue_size, 
                     const std::chrono::steady_clock::time_point& start_time) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "\rProcessed: " << processed 
              << " | Queue: " << queue_size
              << " | Wins: " << global_stats.wins 
              << " | Draws: " << global_stats.draws
              << " | Ongoing: " << global_stats.ongoing
              << " | Unique: " << global_stats.unique_boards
              << " | Time: " << elapsed << "s" << std::flush;
}

// Main generation function
void generate_all_boards() {
    std::cout << "Using " << config.n_workers << " worker threads" << std::endl;
    std::cout << "Chunk size: " << config.chunk_size << " states per iteration" << std::endl;
    std::cout << "Board size: " << config.board_size << "x" << config.board_size << std::endl;
    
    // Add SIZE prefix to filename
    std::string output_filename = config.output_file;
    size_t ext_pos = output_filename.find_last_of('.');
    if (ext_pos != std::string::npos) {
        output_filename = output_filename.substr(0, ext_pos) + "_SIZE" + 
                         std::to_string(config.board_size) + output_filename.substr(ext_pos);
    } else {
        output_filename += "_SIZE" + std::to_string(config.board_size);
    }
    
    std::unordered_set<BoardStr> seen_canonical;
    BoardStateMap board_map(output_filename, config.board_size);
    
    // Initialize with empty board
    Board empty_board(config.board_size, std::string(config.board_size, config.empty_token));
    BoardStr empty_canonical = get_canonical_form(empty_board);
    
    std::vector<BoardStr> current_queue = {empty_canonical};
    seen_canonical.insert(empty_canonical);
    global_stats.unique_boards = 1;
    
    size_t total_processed = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (!current_queue.empty()) {
        // Get chunk to process
        size_t chunk_end = std::min(config.chunk_size, current_queue.size());
        std::vector<BoardStr> process_chunk(
            current_queue.begin(), 
            current_queue.begin() + chunk_end
        );
        current_queue.erase(current_queue.begin(), current_queue.begin() + chunk_end);
        
        std::vector<Board> boards_to_process;
        boards_to_process.reserve(process_chunk.size());
        for (const auto& canonical : process_chunk) {
            boards_to_process.push_back(string_to_board(canonical, config.board_size));
        }
        
        size_t batch_size = std::max(
            (size_t)config.min_batch_size, 
            boards_to_process.size() / (config.n_workers * config.batch_size_divisor)
        );
        
        std::vector<std::vector<Board>> batches;
        for (size_t i = 0; i < boards_to_process.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, boards_to_process.size());
            batches.emplace_back(boards_to_process.begin() + i, boards_to_process.begin() + end);
        }
        
        std::vector<std::thread> threads;
        std::vector<BatchResult> results(batches.size());
        
        for (size_t i = 0; i < batches.size(); i++) {
            threads.emplace_back([&batches, &results, i]() {
                results[i] = process_batch(batches[i]);
            });
            
            // Limit active threads
            if (threads.size() >= (size_t)config.n_workers) {
                for (auto& t : threads) {
                    if (t.joinable()) t.join();
                }
                threads.clear();
            }
        }
        
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }
        
        std::vector<BoardStr> all_new_children;
        for (const auto& result : results) {
            global_stats.wins += result.wins;
            global_stats.draws += result.draws;
            global_stats.ongoing += result.ongoing;
            all_new_children.insert(all_new_children.end(), 
                                   result.new_children.begin(), 
                                   result.new_children.end());
        }
        
        std::vector<BoardStr> unique_new;
        for (const auto& canonical : all_new_children) {
            std::lock_guard<std::mutex> lock(seen_mutex);
            if (seen_canonical.find(canonical) == seen_canonical.end()) {
                seen_canonical.insert(canonical);
                unique_new.push_back(canonical);
                board_map.add_to_buffer(canonical);
            }
        }
        
        global_stats.unique_boards = seen_canonical.size();
        current_queue.insert(current_queue.end(), unique_new.begin(), unique_new.end());
        
        total_processed += process_chunk.size();
        display_progress(total_processed, current_queue.size(), start_time);
    }
    
    board_map.close();
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Total unique boards: " << global_stats.unique_boards << std::endl;
    std::cout << "  Wins: " << global_stats.wins << std::endl;
    std::cout << "  Draws: " << global_stats.draws << std::endl;
    std::cout << "  Ongoing: " << global_stats.ongoing << std::endl;
    std::cout << "Output written to: " << output_filename << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

int main(int argc, char* argv[]) {
    std::string config_file = "config.json";
    if (argc > 1) {
        config_file = argv[1];
    }
    
    std::cout << "Loading configuration from: " << config_file << std::endl;
    load_config(config_file);
    
    std::cout << "\nStarting board generation..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    
    generate_all_boards();
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    
    std::cout << "\nTotal time: " << duration << " seconds" << std::endl;
    
    return 0;
}

