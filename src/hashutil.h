#ifndef CUCKOO_FILTER_HASHUTIL_H_
#define CUCKOO_FILTER_HASHUTIL_H_

#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

#include <string>

#include <openssl/evp.h>
#include <random>

#include <dtl/dtl.hpp>
#include <dtl/filter/blocked_bloomfilter/hash_family.hpp>

namespace cuckoofilter {

class HashUtil {
 public:
  // Bob Jenkins Hash
  static uint32_t BobHash(const void *buf, size_t length, uint32_t seed = 0);
  static uint32_t BobHash(const std::string &s, uint32_t seed = 0);

  // Bob Jenkins Hash that returns two indices in one call
  // Useful for Cuckoo hashing, power of two choices, etc.
  // Use idx1 before idx2, when possible. idx1 and idx2 should be initialized to seeds.
  static void BobHash(const void *buf, size_t length, uint32_t *idx1,
                      uint32_t *idx2);
  static void BobHash(const std::string &s, uint32_t *idx1, uint32_t *idx2);

  // MurmurHash2
  static uint32_t MurmurHash(const void *buf, size_t length, uint32_t seed = 0);
  static uint32_t MurmurHash(const std::string &s, uint32_t seed = 0);

  // SuperFastHash
  static uint32_t SuperFastHash(const void *buf, size_t len);
  static uint32_t SuperFastHash(const std::string &s);

  // Null hash (shift and mask)
  static uint32_t NullHash(const void *buf, size_t length, uint32_t shiftbytes);

  // Wrappers for MD5 and SHA1 hashing using EVP
  static std::string MD5Hash(const char *inbuf, size_t in_length);
  static std::string SHA1Hash(const char *inbuf, size_t in_length);

 private:
  HashUtil();
};


/// Reverse an N-bit quantity in parallel in 5 * lg(N) operations:
/// taken from: https://graphics.stanford.edu/~seander/bithacks.html#ReverseParallel
static uint32_t
reverse_u32(unsigned int v) { // 32-bit word to reverse bit order
  // swap odd and even bits
  v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
  // swap consecutive pairs
  v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
  // swap nibbles ...
  v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
  // swap bytes
  v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
  // swap 2-byte long pairs
  v = ( v >> 16             ) | ( v               << 16);
  return v;
}


// See Martin Dietzfelbinger, "Universal hashing and k-wise independent random
// variables via integer arithmetic without primes".
class TwoIndependentMultiplyShift {
  unsigned __int128 multiply_, add_;

 public:
  TwoIndependentMultiplyShift() {
    ::std::random_device random;
    for (auto v : {&multiply_, &add_}) {
      *v = random();
      for (int i = 1; i <= 4; ++i) {
        *v = *v << 32;
        *v |= random();
      }
    }
  }

  uint64_t operator()(uint64_t key) const {
    return (add_ + multiply_ * static_cast<decltype(multiply_)>(key)) >> 64;
  }
};

class TwoIndependentMultiplyShiftRev {
  unsigned __int128 multiply_, add_;

 public:
  TwoIndependentMultiplyShiftRev() {
    ::std::random_device random;
    for (auto v : {&multiply_, &add_}) {
      *v = random();
      for (int i = 1; i <= 4; ++i) {
        *v = *v << 32;
        *v |= random();
      }
    }
  }

  uint64_t operator()(uint64_t key) const {
    uint64_t h = (add_ + multiply_ * static_cast<decltype(multiply_)>(key)) >> 64;
    uint32_t hi = static_cast<uint32_t>(h >> 32);
    uint32_t lo = static_cast<uint32_t>(h);
    return (static_cast<uint64_t>(reverse_u32(hi)) << 32) | reverse_u32(lo);
  }
};



//struct CRC {
//
//  uint64_t operator()(uint64_t key) const {
//    const uint32_t lo = dtl::hash::stat::mul32<uint32_t, 0>::hash(static_cast<uint32_t>(key));
//    const uint32_t hi = dtl::hash::stat::mul32<uint32_t, 1>::hash(static_cast<uint32_t>(key));
//    return (static_cast<uint64_t>(hi) << 32) | lo;
//  }
//};


struct MultiplyAms {

  uint64_t operator()(uint64_t key) const {
    const uint32_t lo = dtl::hash::stat::mul32<uint32_t, 0>::hash(static_cast<uint32_t>(key));
    const uint32_t hi = dtl::hash::stat::mul32<uint32_t, 1>::hash(static_cast<uint32_t>(key >> 32));
    return (static_cast<uint64_t>(hi) << 32) | lo;
  }
};

struct MultiplyAmsFold {

  uint64_t operator()(uint64_t key) const {
    const uint32_t folded = static_cast<uint32_t>(((key >> 32) ^ 0x85ebca6bu/* Murmur 3 (finalization mix constant)*/) + key);
    // tag
    const uint32_t lo = dtl::hash::stat::mul32<uint32_t, 0>::hash(folded);
    // bucket idx
    const uint32_t hi = dtl::hash::stat::mul32<uint32_t, 1>::hash(folded);
    return (static_cast<uint64_t>(hi) << 32) | lo;
  }
};

struct MultiplyAms64 {

  uint64_t operator()(uint64_t key) const {
    const uint64_t c = (596572387ull /*Peter1*/ << 32) | 370248451ull /*Peter2*/;
    return key * c;
  }
};

struct MultiplyAmsRev {

  uint64_t operator()(uint64_t key) const {
    const uint32_t lo = reverse_u32(dtl::hash::stat::mul32<uint32_t, 0>::hash(static_cast<uint32_t>(key)));
    const uint32_t hi = reverse_u32(dtl::hash::stat::mul32<uint32_t, 1>::hash(static_cast<uint32_t>(key >> 32)));
    return (static_cast<uint64_t>(hi) << 32) | lo;
  }
};

struct Identity {

  uint64_t operator()(uint64_t key) const {
    return key;
  }
};

//template<u64 seed = 0x8445d61a4e774912ull>
struct Murmur64a_64 {

  uint64_t
  operator()(uint64_t key) const {
    const uint64_t m = 0xc6a4a7935bd1e995ull;
    const int r = 47ull;
//    const uint64_t hi = seed ^ (8 /*key length*/ * m);
    const uint64_t hi = 0x8445d61a4e774912ull ^ (8 /*key length*/ * m);
    uint64_t h = hi;
    uint64_t k = key;
    k *= m;
    k ^= k >> r;
    k *= m;
    h ^= k;
    h *= m;
    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
  }
};


// See Patrascu and Thorup's "The Power of Simple Tabulation Hashing"
class SimpleTabulation {
  uint64_t tables_[sizeof(uint64_t)][1 << CHAR_BIT];

 public:
  SimpleTabulation() {
    ::std::random_device random;
    for (unsigned i = 0; i < sizeof(uint64_t); ++i) {
      for (int j = 0; j < (1 << CHAR_BIT); ++j) {
        tables_[i][j] = random() | ((static_cast<uint64_t>(random())) << 32);
      }
    }
  }

  uint64_t operator()(uint64_t key) const {
    uint64_t result = 0;
    for (unsigned i = 0; i < sizeof(key); ++i) {
      result ^= tables_[i][reinterpret_cast<uint8_t *>(&key)[i]];
    }
    return result;
  }
};
}

#endif  // CUCKOO_FILTER_HASHUTIL_H_
