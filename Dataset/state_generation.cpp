#include <iostream>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <string>
#include <sstream>
#include <functional>

struct Config {
    int board_size = 3;
    int n_workers = 1;
    size_t chunk_size = 2000000;
    size_t buffer_size = 5000;
    int batch_size_divisor = 2;
    int min_batch_size = 50;
    std::string output_file = "data/unique_cpp.bin";
};

Config config;

enum CellState : uint8_t {
    EMPTY = 0,
    X_CELL = 1,
    O_CELL = 2
};

struct CompactBoard {
    static constexpr size_t INLINE_CAPACITY = 16; // Inline storage for boards up to 8x8
    
    union {
        uint8_t inline_data[INLINE_CAPACITY];
        uint8_t* heap_data;
    };
    uint16_t size;
    uint16_t capacity;
    
    CompactBoard() : heap_data(nullptr), size(0), capacity(0) {}
    
    CompactBoard(int board_size, int bytes_per_board) : size((uint16_t)board_size), capacity((uint16_t)bytes_per_board) {
        if (bytes_per_board <= INLINE_CAPACITY) {
            std::memset(inline_data, 0, bytes_per_board);
        } else {
            heap_data = new uint8_t[bytes_per_board]();
        }
    }
    
    CompactBoard(const CompactBoard& other) : size(other.size), capacity(other.capacity) {
        if (capacity <= INLINE_CAPACITY) {
            std::memcpy(inline_data, other.inline_data, capacity);
        } else {
            heap_data = new uint8_t[capacity];
            std::memcpy(heap_data, other.heap_data, capacity);
        }
    }
    
    CompactBoard(CompactBoard&& other) noexcept : size(other.size), capacity(other.capacity) {
        if (capacity <= INLINE_CAPACITY) {
            std::memcpy(inline_data, other.inline_data, capacity);
        } else {
            heap_data = other.heap_data;
            other.heap_data = nullptr;
            other.capacity = 0;
        }
        other.size = 0;
    }
    
    CompactBoard& operator=(const CompactBoard& other) {
        if (this != &other) {
            if (capacity > INLINE_CAPACITY) {
                delete[] heap_data;
            }
            size = other.size;
            capacity = other.capacity;
            if (capacity <= INLINE_CAPACITY) {
                std::memcpy(inline_data, other.inline_data, capacity);
            } else {
                heap_data = new uint8_t[capacity];
                std::memcpy(heap_data, other.heap_data, capacity);
            }
        }
        return *this;
    }
    
    CompactBoard& operator=(CompactBoard&& other) noexcept {
        if (this != &other) {
            if (capacity > INLINE_CAPACITY) {
                delete[] heap_data;
            }
            size = other.size;
            capacity = other.capacity;
            if (capacity <= INLINE_CAPACITY) {
                std::memcpy(inline_data, other.inline_data, capacity);
            } else {
                heap_data = other.heap_data;
                other.heap_data = nullptr;
                other.capacity = 0;
            }
            other.size = 0;
        }
        return *this;
    }
    
    ~CompactBoard() {
        if (capacity > INLINE_CAPACITY) {
            delete[] heap_data;
        }
    }
    
    inline uint8_t* data_ptr() {
        return (capacity <= INLINE_CAPACITY) ? inline_data : heap_data;
    }
    
    inline const uint8_t* data_ptr() const {
        return (capacity <= INLINE_CAPACITY) ? inline_data : heap_data;
    }

    inline void set_cell(int row, int col, CellState state) {
        int cell_index = row * size + col;
        int bit_position = cell_index * 2;
        int byte_index = bit_position >> 3;
        int bit_offset = bit_position & 7;

        uint8_t* data = data_ptr();
        uint8_t mask = uint8_t(3u << bit_offset);
        uint8_t val = uint8_t(uint8_t(state) << bit_offset);

        data[byte_index] = (data[byte_index] & ~mask);
        data[byte_index] = (data[byte_index] | val);

        if (bit_offset > 6) {
            uint8_t nextMask = 1u;
            uint8_t nextVal = uint8_t(uint8_t(state) >> 1);
            data[byte_index + 1] = (data[byte_index + 1] & ~nextMask);
            data[byte_index + 1] = (data[byte_index + 1] | nextVal);
        }
    }

    inline CellState get_cell(int row, int col) const {
        int cell_index = row * size + col;
        int bit_position = cell_index * 2;
        int byte_index = bit_position >> 3;
        int bit_offset = bit_position & 7;

        const uint8_t* data = data_ptr();
        uint8_t value = (data[byte_index] >> bit_offset) & 3u;
        if (bit_offset > 6) {
            value |= (data[byte_index + 1] & 1u) << 1;
        }
        return static_cast<CellState>(value);
    }

    bool operator==(const CompactBoard& other) const noexcept {
        if (size != other.size || capacity != other.capacity) return false;
        return std::memcmp(data_ptr(), other.data_ptr(), capacity) == 0;
    }

    bool operator<(const CompactBoard& other) const noexcept {
        if (size != other.size) return size < other.size;
        if (capacity != other.capacity) return capacity < other.capacity;
        return std::memcmp(data_ptr(), other.data_ptr(), capacity) < 0;
    }
};

struct CompactBoardHash {
    size_t operator()(const CompactBoard& b) const noexcept {
        // 64-bit FNV-1a
        const uint64_t fnv_offset = 14695981039346656037ull;
        const uint64_t fnv_prime  = 1099511628211ull;
        uint64_t h = fnv_offset;
        const uint8_t* data = b.data_ptr();
        for (size_t i = 0; i < b.capacity; ++i) {
            h ^= (uint64_t)data[i];
            h *= fnv_prime;
        }
        h ^= (uint64_t)b.size;
        h *= fnv_prime;
        return (size_t)h;
    }
};

struct Statistics {
    std::atomic<size_t> wins{0};
    std::atomic<size_t> draws{0};
    std::atomic<size_t> ongoing{0};
    std::atomic<size_t> unique_boards{0};
} global_stats;

std::mutex cout_mutex;
std::mutex file_mutex;
std::mutex seen_mutex;

// New atomic for processed counter
std::atomic<size_t> total_processed_atomic{0};

static inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r\"");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r\",");
    if (last == std::string::npos) last = str.size() - 1;
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
        else if (key == "n_workers") config.n_workers = std::max(1, std::stoi(value));
        else if (key == "chunk_size") config.chunk_size = std::stoull(value);
        else if (key == "buffer_size") config.buffer_size = std::stoull(value);
        else if (key == "batch_size_divisor") config.batch_size_divisor = std::stoi(value);
        else if (key == "min_batch_size") config.min_batch_size = std::stoi(value);
        else if (key == "output_file") config.output_file = value;
    }
}

CompactBoard rotate_90(const CompactBoard& board, int bytes_per_board) {
    int n = board.size;
    CompactBoard rotated(n, bytes_per_board);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            rotated.set_cell(i, j, board.get_cell(n - 1 - j, i));
        }
    }
    return rotated;
}

CompactBoard flip_horizontal(const CompactBoard& board, int bytes_per_board) {
    int n = board.size;
    CompactBoard flipped(n, bytes_per_board);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            flipped.set_cell(i, j, board.get_cell(i, n - 1 - j));
        }
    }
    return flipped;
}

CompactBoard get_canonical_form(const CompactBoard& board, int bytes_per_board) {
    std::vector<CompactBoard> variants;
    variants.reserve(8);

    CompactBoard cur = board;
    for (int k = 0; k < 4; ++k) {
        variants.emplace_back(std::move(cur));
        cur = rotate_90(variants.back(), bytes_per_board); // rotate last element
    }
    CompactBoard flipped = flip_horizontal(board, bytes_per_board);
    cur = flipped;
    for (int k = 0; k < 4; ++k) {
        variants.emplace_back(std::move(cur));
        cur = rotate_90(variants.back(), bytes_per_board);
    }

    CompactBoard const* best = &variants[0];
    for (size_t i = 1; i < variants.size(); ++i) {
        if (variants[i] < *best) best = &variants[i];
    }
    return *best;
}

inline bool check_line_run(const CompactBoard& board, int win_len, int start_r, int start_c, int dr, int dc, int steps) {
    uint8_t prev = 0;
    int count = 0;
    for (int s = 0; s < steps; ++s) {
        int r = start_r + s * dr;
        int c = start_c + s * dc;
        uint8_t cell = uint8_t(board.get_cell(r, c));
        if (cell == 0) {
            prev = 0;
            count = 0;
            continue;
        }
        if (cell == prev) {
            ++count;
            if (count >= win_len) return true;
        } else {
            prev = cell;
            count = 1;
        }
    }
    return false;
}

bool is_board_ended(const CompactBoard& board) {
    int n = board.size;
    int win_len = std::min(n, 5);

    // rows
    for (int r = 0; r < n; ++r) {
        if (check_line_run(board, win_len, r, 0, 0, 1, n)) return true;
    }
    // cols
    for (int c = 0; c < n; ++c) {
        if (check_line_run(board, win_len, 0, c, 1, 0, n)) return true;
    }
    // diagonals TL-BR
    for (int start = 0; start <= 2 * (n - 1); ++start) {
        int r0 = std::max(0, start - (n - 1));
        int c0 = start - r0;
        int steps = std::min(n - r0, c0 + 1);
        if (steps >= win_len) {
            if (check_line_run(board, win_len, r0, c0, 1, -1, steps)) return true;
        }
    }
    // diagonals TR-BL
    for (int start = - (n - 1); start <= (n - 1); ++start) {
        int r0 = std::max(0, start);
        int c0 = r0 - start;
        int steps = std::min(n - r0, n - c0);
        if (steps >= win_len) {
            if (check_line_run(board, win_len, r0, c0, 1, 1, steps)) return true;
        }
    }

    return false;
}

bool is_board_full(const CompactBoard& board) {
    int n = board.size;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (board.get_cell(i, j) == EMPTY) return false;
    return true;
}

std::pair<int,int> count_tokens(const CompactBoard& board) {
    int x = 0, o = 0;
    int n = board.size;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            auto v = board.get_cell(i, j);
            if (v == X_CELL) ++x;
            else if (v == O_CELL) ++o;
        }
    return {x, o};
}

// Batch result: lightweight
struct BatchResult {
    size_t wins = 0;
    size_t draws = 0;
    size_t ongoing = 0;
    std::vector<CompactBoard> new_children;
};

// Process a batch producing canonical children but dedup within this batch
BatchResult process_batch(const std::vector<CompactBoard>& boards, int bytes_per_board) {
    BatchResult res;
    std::unordered_set<CompactBoard, CompactBoardHash> local_set;
    local_set.reserve(boards.size() * 4 + 16);

    for (const auto& board : boards) {
        bool has_winner = is_board_ended(board);
        bool full = is_board_full(board);
        if (has_winner || full) {
            if (has_winner) ++res.wins;
            else ++res.draws;
            continue;
        }
        ++res.ongoing;
        auto [xc, oc] = count_tokens(board);
        CellState turn = (xc == oc) ? X_CELL : O_CELL;
        int n = board.size;

        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                if (board.get_cell(r, c) != EMPTY) continue;
                CompactBoard child = board;
                child.set_cell(r, c, turn);
                CompactBoard canonical = get_canonical_form(child, bytes_per_board);

                auto it = local_set.find(canonical);
                if (it == local_set.end()) {
                    local_set.emplace(std::move(canonical));
                }
            }
        }
    }

    res.new_children.reserve(local_set.size());
    for (auto &b : local_set) {
        res.new_children.push_back(std::move(const_cast<CompactBoard&>(b)));
    }
    return res;
}

// BoardStateMap: writes binary boards in buffered manner (single large byte buffer)
class BoardStateMap {
    std::ofstream file;
    int board_size;
    int bytes_per_board;
    std::vector<uint8_t> write_buffer;
public:
    BoardStateMap(const std::string &filename, int size, int bytesPerBoard)
        : board_size(size), bytes_per_board(bytesPerBoard)
    {
        file.open(filename, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) throw std::runtime_error("Cannot open output file");
        uint32_t header[2] = {static_cast<uint32_t>(board_size), static_cast<uint32_t>(bytes_per_board)};
        file.write(reinterpret_cast<const char*>(header), sizeof(header));
        write_buffer.reserve((size_t)bytes_per_board * std::min<size_t>(config.buffer_size, 1024));
    }

    void append_board_binary(const CompactBoard &b) {
        const uint8_t* ptr = b.data_ptr();
        write_buffer.insert(write_buffer.end(), ptr, ptr + bytes_per_board);
        if (write_buffer.size() >= bytes_per_board * config.buffer_size) {
            flush();
        }
    }

    void flush() {
        if (write_buffer.empty()) return;
        file.write(reinterpret_cast<const char*>(write_buffer.data()), write_buffer.size());
        write_buffer.clear();
    }

    void close() {
        flush();
        if (file.is_open()) file.close();
    }

    ~BoardStateMap() {
        if (file.is_open()) close();
    }
};

void display_progress(size_t processed, size_t queue_size,
                      const std::chrono::steady_clock::time_point& start_time)
{
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "\rProcessed: " << processed
              << " | Queue: " << queue_size
              << " | Wins: " << global_stats.wins
              << " | Draws: " << global_stats.draws
              << " | Ongoing: " << global_stats.ongoing
              << " | Unique: " << global_stats.unique_boards
              << " | Time: " << elapsed << "s"
              << std::flush;
}

void generate_all_boards() {
    std::cout << "Memory-priority generation\n";
    std::cout << "Workers: " << config.n_workers << "\n";
    std::cout << "Board size: " << config.board_size << "x" << config.board_size << "\n";

    int cells = config.board_size * config.board_size;
    int bits_per_board = cells * 2;
    int bytes_per_board = (bits_per_board + 7) / 8;
    std::cout << "Bytes per board: " << bytes_per_board << "\n";

    std::string output_filename = config.output_file;
    size_t pos = output_filename.find_last_of('.');
    if (pos != std::string::npos) {
        output_filename = output_filename.substr(0, pos) + "_SIZE" + std::to_string(config.board_size) + output_filename.substr(pos);
    } else {
        output_filename += "_SIZE" + std::to_string(config.board_size);
    }

    std::unordered_set<CompactBoard, CompactBoardHash> seen;
    seen.reserve(1024);

    BoardStateMap board_map(output_filename, config.board_size, bytes_per_board);

    CompactBoard empty(config.board_size, bytes_per_board);
    CompactBoard empty_canonical = get_canonical_form(empty, bytes_per_board);

    std::vector<CompactBoard> current_queue;
    current_queue.reserve(1024);
    current_queue.push_back(empty_canonical);
    seen.emplace(empty_canonical);
    global_stats.unique_boards = seen.size();

    // prepare layer printed flags (0..cells)
    std::vector<char> layer_printed((size_t)cells + 1, 0);

    // use atomic processed counter
    total_processed_atomic.store(0);

    auto start_time = std::chrono::steady_clock::now();

    // print layer 0 immediately (empty board)
    {
        size_t unique_now = seen.size();
        size_t processed_now = total_processed_atomic.load();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start_time).count();
        {
            std::lock_guard<std::mutex> cout_lock(cout_mutex);
            std::cout << std::endl;
            std::cout << "[Layer 0] reached — Unique: " << unique_now
                      << " | Processed: " << processed_now
                      << " | Time: " << elapsed << "s" << std::endl;
        }
        layer_printed[0] = 1;
    }

    while (!current_queue.empty()) {
        size_t take = std::min(config.chunk_size, current_queue.size());
        std::vector<CompactBoard> chunk;
        chunk.reserve(take);
        for (size_t i = 0; i < take; ++i) {
            chunk.push_back(std::move(current_queue[i]));
        }
        current_queue.erase(current_queue.begin(), current_queue.begin() + take);

        size_t batch_size = std::max((size_t)config.min_batch_size, chunk.size() / ( (size_t)config.n_workers * (size_t)config.batch_size_divisor ));
        if (batch_size == 0) batch_size = chunk.size();

        std::vector<std::pair<size_t,size_t>> batch_ranges;
        for (size_t i = 0; i < chunk.size(); i += batch_size) {
            size_t end = std::min(chunk.size(), i + batch_size);
            batch_ranges.emplace_back(i, end);
        }

        std::vector<BatchResult> results(batch_ranges.size());
        std::vector<std::thread> threads;
        threads.reserve(batch_ranges.size());

        for (size_t bi = 0; bi < batch_ranges.size(); ++bi) {
            size_t bstart = batch_ranges[bi].first;
            size_t bend   = batch_ranges[bi].second;
            threads.emplace_back([bstart,bend,&chunk,&results,bi,bytes_per_board]() {
                std::vector<CompactBoard> view;
                view.reserve(bend - bstart);
                for (size_t t = bstart; t < bend; ++t) view.push_back(chunk[t]);
                results[bi] = process_batch(view, bytes_per_board);
            });

            if (threads.size() >= (size_t)config.n_workers) {
                for (auto &t : threads) if (t.joinable()) t.join();
                threads.clear();
            }
        }
        for (auto &t : threads) if (t.joinable()) t.join();

        std::vector<CompactBoard> new_unique_batch;
        size_t batch_wins = 0, batch_draws = 0, batch_ongoing = 0;
        size_t approx_new_children = 0;
        for (auto &r : results) approx_new_children += r.new_children.size();
        new_unique_batch.reserve(approx_new_children);

        std::unordered_set<CompactBoard, CompactBoardHash> local_union;
        local_union.reserve(approx_new_children + 16);

        for (auto &r : results) {
            batch_wins += r.wins;
            batch_draws += r.draws;
            batch_ongoing += r.ongoing;
            for (auto &c : r.new_children) {
                local_union.emplace(std::move(c));
            }
        }

        {
            std::lock_guard<std::mutex> lock(seen_mutex);
            for (auto &cand : local_union) {
                auto insert_result = seen.emplace(cand);
                if (insert_result.second) {
                    new_unique_batch.push_back(cand);
                    board_map.append_board_binary(cand);

                    auto [xc, oc] = count_tokens(cand);
                    int filled = xc + oc;
                    if (filled >= 0 && filled <= cells && !layer_printed[filled]) {
                        // mark and print once per layer
                        layer_printed[filled] = 1;
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                        size_t unique_now = seen.size();
                        size_t processed_now = total_processed_atomic.load();
                        std::lock_guard<std::mutex> cout_lock(cout_mutex);
                        std::cout << std::endl;
                        std::cout << "[Layer " << filled << "] reached — Unique: " << unique_now
                                  << " | Processed: " << processed_now
                                  << " | Time: " << elapsed << "s" << std::endl;
                    }
                }
            }
            global_stats.unique_boards = seen.size();
        }

        global_stats.wins += batch_wins;
        global_stats.draws += batch_draws;
        global_stats.ongoing += batch_ongoing;

        if (!new_unique_batch.empty()) {
            current_queue.insert(current_queue.end(), std::make_move_iterator(new_unique_batch.begin()), std::make_move_iterator(new_unique_batch.end()));
        }

        total_processed_atomic.fetch_add(take);
        display_progress(total_processed_atomic.load(), current_queue.size(), start_time);
    }

    board_map.close();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Total unique boards: " << global_stats.unique_boards << std::endl;
    std::cout << "  Wins: " << global_stats.wins << std::endl;
    std::cout << "  Draws: " << global_stats.draws << std::endl;
    std::cout << "  Ongoing: " << global_stats.ongoing << std::endl;
    std::cout << "File written to: " << output_filename << " (binary format)\n";
    std::cout << std::string(60, '=') << std::endl;
}

int main(int argc, char* argv[]) {
    std::string config_file = "Dataset/config.json";
    if (argc > 1) config_file = argv[1];

    std::cout << "Loading configuration from: " << config_file << std::endl;
    load_config(config_file);

    std::cout << "Starting board generation (Memory-optimized)..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    generate_all_boards();
    auto end = std::chrono::steady_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "Total time: " << dur << " seconds\n";
    return 0;
}
