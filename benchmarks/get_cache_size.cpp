#include <cstring>
#include <fstream>

#include <dtl/dtl.hpp>

//===----------------------------------------------------------------------===//
// Helper to determine the cache sizes.
// Used by the benchmark script.
//===----------------------------------------------------------------------===//

u1
file_exists(const std::string& name) {
  std::ifstream f(name.c_str());
  return f.good();
}

std::vector<uint64_t>
get_cache_sizes() {
  // WARNING: non-portable code
  const std::string sys_path = "/sys/devices/system/cpu/cpu0/cache";
  std::vector<$u64> cache_sizes;

  $i64 idx = -1;
  while(true) {
    idx++;
    const std::string type_filename = sys_path + "/index" + std::to_string(idx) + "/type";
    const std::string size_filename = sys_path + "/index" + std::to_string(idx) + "/size";
    if (!file_exists(type_filename) || !file_exists(size_filename)) break;

    std::ifstream type_file(type_filename);
    std::string type_str;
    std::getline(type_file, type_str);
    type_file.close();
    if (type_str == "Instruction") continue; // skip instruction cache

    std::ifstream size_file(size_filename);
    std::string size_str;
    std::getline(size_file, size_str);
    size_file.close();

    $u64 cache_size = 0;

    // parsing inspired from: http://www.cs.columbia.edu/~orestis/vbf.c
    auto iter = size_str.begin();
    while (iter != size_str.end() && isdigit(*iter)) {
      cache_size = (cache_size * 10) + *iter++ - '0';
    }
    if (iter != size_str.end()) {
      const auto unit = *iter;
      switch (unit) {
        case 'K': cache_size <<= 10; break;
        case 'M': cache_size <<= 20; break;
        case 'G': cache_size <<= 30; break;
        default:
          throw std::runtime_error("Failed to parse cache size: " + size_str);
      }
    }
    cache_sizes.push_back(cache_size);
  }

  if (cache_sizes.size() == 0) {
    throw std::runtime_error("Failed to determine the cache sizes.");
  }
  return cache_sizes;
}

int main(int argc, char* argv[]) {

  const auto cache_sizes = get_cache_sizes();

  if (argc > 1) {

    uint64_t level = 0;
    try {
      level = std::stoll(argv[1]);
    } catch (...) {
      std::cerr << "Illegal argument." << std::endl;
      return 1;
    }
    if (level == 0 || (level - 1) >= cache_sizes.size()) {
      std::cerr << "Illegal argument. Unknown cache level." << std::endl;
      return 1;
    }
    std::cout << cache_sizes[level - 1] << std::endl;
    return 0;
  }
  else {
    // Print the number of cache levels.
    std::cout << cache_sizes.size() << std::endl;
    return 0;
  }
}