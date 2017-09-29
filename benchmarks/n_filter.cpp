#include <algorithm>
#include <cstdlib>
#include <set>
#include <vector>

#include <dtl/dtl.hpp>
#include <dtl/env.hpp>
#include <dtl/thread.hpp>

//===----------------------------------------------------------------------===//
// Helper to filter the result data.
// Used by the benchmark script.
//===----------------------------------------------------------------------===//

int main(int argc, char * argv[]) {

  //===----------------------------------------------------------------------===//
  // Max filter size in bits
  u64 m_hi = std::min(dtl::env<$u64>::get("M_HI", u64(256ull) * 1024 * 1024 * 8),
      u64(256ull) * 1024 * 1024 * 8); // at most 256 MiB
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Bits-per-element rate
  // repeats the benchmark with different bits-per-element rates
  const auto b_lo = dtl::env<$u64>::get("BITS_PER_ELEMENT_LO", 4);
  //===----------------------------------------------------------------------===//

  std::this_thread::sleep_for(std::chrono::seconds {1});
  u64 n_lo = dtl::next_power_of_two(dtl::env<$u64>::get("N_LO", 1ull << 10));
  u64 n_lo_log2 = dtl::log_2(n_lo);
  u64 n_hi = dtl::next_power_of_two(dtl::env<$u64>::get("N_HI", 1ull << 26));
  u64 n_hi_log2 = dtl::log_2(n_hi);

  f64 q = dtl::env<$f64>::get("Q", 0.0);


  // all valid n's
  const std::set<$u64> n_s = [&]() {
    std::set<$u64> n_s;

    for ($u64 n_log2 = n_lo_log2; n_log2 <= n_hi_log2; n_log2++) {
//      const std::vector<$f64> exp {
//          n_log2 +  0 * 0.0625,
//          n_log2 +  1 * 0.0625,
//          n_log2 +  2 * 0.0625,
//          n_log2 +  3 * 0.0625,
//          n_log2 +  4 * 0.0625,
//          n_log2 +  5 * 0.0625,
//          n_log2 +  6 * 0.0625,
//          n_log2 +  7 * 0.0625,
//          n_log2 +  8 * 0.0625,
//          n_log2 +  9 * 0.0625,
//          n_log2 + 10 * 0.0625,
//          n_log2 + 11 * 0.0625,
//          n_log2 + 12 * 0.0625,
//          n_log2 + 13 * 0.0625,
//          n_log2 + 14 * 0.0625,
//          n_log2 + 15 * 0.0625,
//          n_log2 + 16 * 0.0625,
//      };
//      const std::vector<$f64> exp {
//          n_log2 + 0 * 0.125,
//          n_log2 + 1 * 0.125,
//          n_log2 + 2 * 0.125,
//          n_log2 + 3 * 0.125,
//          n_log2 + 4 * 0.125,
//          n_log2 + 5 * 0.125,
//          n_log2 + 6 * 0.125,
//          n_log2 + 7 * 0.125,
//          n_log2 + 8 * 0.125,
//      };
      const std::vector<$f64> exp {
          n_log2 + 0 * 0.25,
          n_log2 + 1 * 0.25,
          n_log2 + 2 * 0.25,
          n_log2 + 3 * 0.25,
          n_log2 + 4 * 0.25,
      };

      for (auto e : exp) {
        u64 n = std::pow(2, e);
        if ((n * b_lo) > m_hi) continue;
        n_s.insert(n);
      }
    }
    return n_s;
  }();


  if (q <= 1.0) {
    // print all n's
    for (auto n : n_s) {
      std::cout << n << " " << std::endl;
    }
  }
  else {
    for ($u64 n_log2 = n_lo_log2; n_log2 <= n_hi_log2; n_log2++) {
      i64 s = (1ull << n_log2) * q;
      auto lower = std::lower_bound(n_s.begin(), n_s.end(), s);
      auto upper = std::upper_bound(n_s.begin(), n_s.end(), s);
      auto d_l = std::abs(s - static_cast<i64>(*lower));
      auto d_u = std::abs(s - static_cast<i64>(*upper));
      std::cout << (d_l < d_u ? *lower : *upper) << std::endl;
    }
  }

  return 0;
}
