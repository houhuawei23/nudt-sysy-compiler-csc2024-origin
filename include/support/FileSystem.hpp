#pragma once
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

namespace utils {
void ensure_directory_exists(const std::string& path) {
  fs::path dir_path(path);
  if (!fs::exists(dir_path)) {
    if (fs::create_directories(dir_path)) {
      std::cout << "Directory created: " << path << std::endl;
    } else {
      std::cerr << "Failed to create directory: " << path << std::endl;
    }
  } else {
    std::cout << "Directory already exists: " << path << std::endl;
  }
}

}  // namespace utils