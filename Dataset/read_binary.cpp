#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdint>

enum CellState : uint8_t {
    EMPTY = 0,
    X_CELL = 1,
    O_CELL = 2
};

CellState get_cell(const std::vector<uint8_t>& data, int board_size, int row, int col) {
    int cell_index = row * board_size + col;
    int bit_position = cell_index * 2;
    int byte_index = bit_position / 8;
    int bit_offset = bit_position % 8;
    
    uint8_t value = (data[byte_index] >> bit_offset) & 3;
    
    if (bit_offset == 7) {
        value |= (data[byte_index + 1] & 1) << 1;
    }
    
    return static_cast<CellState>(value);
}

void print_board(const std::vector<uint8_t>& data, int board_size) {
    for (int i = 0; i < board_size; i++) {
        for (int j = 0; j < board_size; j++) {
            CellState state = get_cell(data, board_size, i, j);
            char c;
            switch (state) {
                case EMPTY: c = '-'; break;
                case X_CELL: c = 'X'; break;
                case O_CELL: c = 'O'; break;
                default: c = '?'; break;
            }
            std::cout << c;
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <binary_file> [num_boards_to_display]" << std::endl;
        std::cout << "\nReads and displays board states from binary format files." << std::endl;
        std::cout << "Binary format: 2 bits per cell (EMPTY=0, X=1, O=2)" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    int num_to_display = (argc > 2) ? std::stoi(argv[2]) : 10;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return 1;
    }
    
    // Read header
    uint32_t header[2];
    file.read(reinterpret_cast<char*>(header), sizeof(header));
    
    int board_size = header[0];
    int bytes_per_board = header[1];
    
    std::cout << "File: " << filename << std::endl;
    std::cout << "Board size: " << board_size << "x" << board_size << std::endl;
    std::cout << "Bytes per board: " << bytes_per_board << std::endl;
    std::cout << "Bits per board: " << (board_size * board_size * 2) << std::endl;
    
    // Count total boards
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    size_t total_boards = (file_size - 8) / bytes_per_board;
    
    std::cout << "Total boards: " << total_boards << std::endl;
    std::cout << "File size: " << file_size << " bytes" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Go back to start of data
    file.seekg(8, std::ios::beg);
    
    // Read and display boards
    std::vector<uint8_t> board_data(bytes_per_board);
    int displayed = 0;
    
    for (size_t i = 0; i < total_boards && displayed < num_to_display; i++) {
        file.read(reinterpret_cast<char*>(board_data.data()), bytes_per_board);
        
        std::cout << "\nBoard #" << (i + 1) << ":" << std::endl;
        print_board(board_data, board_size);
        
        // Print raw binary data
        std::cout << "Binary: ";
        for (int j = 0; j < bytes_per_board; j++) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') 
                     << static_cast<int>(board_data[j]) << " ";
        }
        std::cout << std::dec << std::endl;
        
        displayed++;
    }
    
    if (total_boards > num_to_display) {
        std::cout << "\n... (" << (total_boards - num_to_display) << " more boards)" << std::endl;
    }
    
    file.close();
    return 0;
}

