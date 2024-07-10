#ifndef SPM_CSV_H
#define SPM_CSV_H

#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

template <typename T>
void writeCSV(const std::string& filename, const std::vector<std::string>& headers, const std::vector<std::vector<T>>& data) {
    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path resultsPath = currentPath / "results";
    std::filesystem::path filePath = resultsPath / filename;

    std::ofstream file(filePath);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    // Write headers
    for (size_t i = 0; i < headers.size(); ++i) {
        file << headers[i];
        if (i < headers.size() - 1) {
            file << ",";
        }
    }
    file << "\n";

    // Write data
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << std::to_string(row[i]);
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}

#endif //SPM_CSV_H
