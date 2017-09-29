#pragma once

#if defined(__AVX2__)

#include <dtl/dtl.hpp>
#include <dtl/math.hpp>

namespace columbia {

//===----------------------------------------------------------------------===//
// C++ wrapper for the Columbia Vectorized Bloom filter.
//===----------------------------------------------------------------------===//
class vbf {

  /// The (actual) bit length of the Bloom filter.
  std::size_t m;

  /// log_2(m)
  std::size_t m_log2;

  /// 32 - log_2(m)
  uint8_t shift;

  /// The number of hash functions.
  u32 k;


public:

  using key_t = $u32;
  using word_t = $u32;

  //===----------------------------------------------------------------------===//
  // The API functions.
  //===----------------------------------------------------------------------===//
  void
  insert(word_t* __restrict filter_data, key_t key);

  void
  batch_insert(word_t* __restrict filter_data, const key_t* keys, u32 key_cnt);

  $u1
  contains(const word_t* __restrict filter_data, key_t key) const;

  $u64
  batch_contains(const word_t* __restrict filter_data,
                 const key_t* keys, u32 key_cnt,
                 $u32* match_positions, u32 match_offset) const;

  std::string
  name() const;

  std::size_t
  size_in_bytes() const;

  std::size_t
  size() const;

  //===----------------------------------------------------------------------===//

  vbf(std::size_t m, u32 k)
      : m(dtl::next_power_of_two(m)),
        m_log2(dtl::log_2(dtl::next_power_of_two(m))),
        shift(32 - dtl::log_2(dtl::next_power_of_two(m))),
        k(k) {}
  ~vbf();
  vbf(vbf&&);
  vbf(const vbf&) = delete;
  vbf& operator=(vbf&&);
//  vbf& operator=(const vbf&) = delete;

};

} // namespace columbia

#endif //defined(__AVX2__)