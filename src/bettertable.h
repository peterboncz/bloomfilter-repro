/* derived from 'singletable.h' */
#pragma once

#include <assert.h>

#include <sstream>

#include <xmmintrin.h>

#include "bitsutil.h"
#include "debug.h"
#include "printutil.h"

#include <dtl/thread.hpp>

namespace cuckoofilter {

// the most naive table implementation: one huge bit array
template <size_t bits_per_tag, size_t tags_per_bucket>
class BetterTable {
 public:
  using word_t = char;

  static constexpr size_t kTagsPerBucket = tags_per_bucket;
  static constexpr size_t kBytesPerBucket =
      (bits_per_tag * kTagsPerBucket + 7) >> 3;
  static constexpr uint32_t kTagMask = (1ULL << bits_per_tag) - 1;
  static constexpr bool delete_supported = false;
  static constexpr bool counting_supported = false; // a counting cuckoo table stores duplicates
  static_assert(!delete_supported || counting_supported, "Deletion is only supported in combination with counting.");

  struct Bucket {
    char bits_[kBytesPerBucket];
  } __attribute__((__packed__));

  // using a pointer adds one more indirection
//  Bucket *buckets_;
  const size_t num_buckets_;

 public:
  explicit BetterTable(const size_t num) : num_buckets_(num) {
//    std::cout << "num buckets: " << num << std::endl;
  }

  ~BetterTable() { }

  size_t BitsPerTag() const {
    return bits_per_tag;
  }

  size_t NumBuckets() const {
    return num_buckets_;
  }

  size_t SizeInBytes() const { 
    return kBytesPerBucket * num_buckets_; 
  }

  size_t SizeInTags() const { 
    return kTagsPerBucket * num_buckets_; 
  }

  std::string Info() const {
    std::stringstream ss;
    ss << "Cuckoo table (BetterTable):\n";
    ss << "\t\tTag size: " << bits_per_tag << " bits \n";
    ss << "\t\tAssociativity: " << kTagsPerBucket << "\n";
    ss << "\t\tTotal # of buckets: " << num_buckets_ << "\n";
    ss << "\t\tTotal # slots: " << SizeInTags() << "\n";
    ss << "\t\tBytes per bucket: " << kBytesPerBucket << "\n";

    return ss.str();
  }

  // read tag from pos(i,j)
  inline uint32_t ReadTag(const word_t* __restrict filter_data, const size_t i, const size_t j) const {
//    const auto* buckets = reinterpret_cast<const Bucket*>(filter_data);
//    const char* p = buckets[i].bits_;
    const auto pos = i * sizeof(Bucket);
    const char* p = &filter_data[pos];
    uint32_t tag;
    /* following code only works for little-endian */
    if (bits_per_tag == 2) {
      tag = *((uint8_t *)p) >> (j * 2);
    }
    else if (bits_per_tag == 4) {
      p += (j >> 1);
      tag = *((uint8_t *)p) >> ((j & 1) << 2);
    }
    else if (bits_per_tag == 8) {
      p += j;
      tag = *((uint8_t *)p);
    }
    else if (bits_per_tag == 12) {
      p += j + (j >> 1);
      tag = *((uint16_t *)p) >> ((j & 1) << 2);
    }
    else if (bits_per_tag == 16) {
      p += (j << 1);
      tag = *((uint16_t *)p);
    }
    else if (bits_per_tag == 32) {
      tag = ((uint32_t *)p)[j];
    }
    return tag & kTagMask;
  }

  // write tag to pos(i,j)
  inline void WriteTag(word_t* __restrict filter_data,
                       const size_t i, const size_t j, const uint32_t t) const {
    assert(i < num_buckets_);
    assert(j < kTagsPerBucket);
    assert((t & (~kTagMask)) == 0);
    auto* buckets = reinterpret_cast<Bucket*>(filter_data);
    char *p = buckets[i].bits_;
    uint32_t tag = t & kTagMask;
    /* following code only works for little-endian */
    if (bits_per_tag == 2) {
      *((uint8_t *)p) |= tag << (2 * j);
    }
    else if (bits_per_tag == 4) {
      p += (j >> 1);
      if ((j & 1) == 0) {
        *((uint8_t *)p) &= 0xf0;
        *((uint8_t *)p) |= tag;
      }
      else {
        *((uint8_t *)p) &= 0x0f;
        *((uint8_t *)p) |= (tag << 4);
      }
    }
    else if (bits_per_tag == 8) {
      ((uint8_t *)p)[j] = tag;
    }
    else if (bits_per_tag == 12) {
      p += (j + (j >> 1));
      if ((j & 1) == 0) {
        ((uint16_t *)p)[0] &= 0xf000;
        ((uint16_t *)p)[0] |= tag;
      } else {
        ((uint16_t *)p)[0] &= 0x000f;
        ((uint16_t *)p)[0] |= (tag << 4);
      }
    }
    else if (bits_per_tag == 16) {
      ((uint16_t *)p)[j] = tag;
    }
    else if (bits_per_tag == 32) {
      ((uint32_t *)p)[j] = tag;
    }
  }

  inline bool FindTagInBuckets(const word_t* __restrict filter_data,
                               const size_t i1, const size_t i2,
                               const uint32_t tag) const {
    const auto* buckets = reinterpret_cast<const Bucket*>(filter_data);
    const char *p1 = buckets[i1].bits_;
    const char *p2 = buckets[i2].bits_;

    uint64_t v1 = *((uint64_t *)p1);
    uint64_t v2 = *((uint64_t *)p2);

    // caution: unaligned access & assuming little endian
    if (bits_per_tag == 4 && kTagsPerBucket == 4) {
      return hasvalue4(v1, tag) | hasvalue4(v2, tag);
    }
    else if (bits_per_tag == 8 && kTagsPerBucket == 4) {
      return hasvalue8(v1, tag) | hasvalue8(v2, tag);
    }
    else if (bits_per_tag == 12 && kTagsPerBucket == 4) {
      return hasvalue12(v1, tag) | hasvalue12(v2, tag);
    }
    else if (bits_per_tag == 16 && kTagsPerBucket == 4) {
      return hasvalue16(v1, tag) | hasvalue16(v2, tag);
    }
    else {
      for (size_t j = 0; j < kTagsPerBucket; j++) {
        if ((ReadTag(filter_data, i1, j) == tag) || (ReadTag(filter_data, i2, j) == tag)) {
          return true;
        }
      }
      return false;
    }
  }

//  inline bool FindTagInBucket(const word_t* __restrict filter_data,
//                              const size_t i, const uint32_t tag) const {
//    const auto* buckets = reinterpret_cast<const Bucket*>(filter_data);
//    // caution: unaligned access & assuming little endian
//    if (bits_per_tag == 4 && kTagsPerBucket == 4) {
//      const char *p = buckets[i].bits_;
//      uint64_t v = *(uint64_t *)p;  // uint16_t may suffice
//      return hasvalue4(v, tag);
//    } else if (bits_per_tag == 8 && kTagsPerBucket == 4) {
//      const char *p = buckets[i].bits_;
//      uint64_t v = *(uint64_t *)p;  // uint32_t may suffice
//      return hasvalue8(v, tag);
//    } else if (bits_per_tag == 12 && kTagsPerBucket == 4) {
//      const char *p = buckets[i].bits_;
//      uint64_t v = *(uint64_t *)p;
//      return hasvalue12(v, tag);
//    } else if (bits_per_tag == 16 && kTagsPerBucket == 4) {
//      const char *p = buckets[i].bits_;
//      uint64_t v = *(uint64_t *)p;
//      return hasvalue16(v, tag);
//    } else {
//      for (size_t j = 0; j < kTagsPerBucket; j++) {
//        if (ReadTag(filter_data, i, j) == tag) {
//          return true;
//        }
//      }
//      return false;
//    }
//  }

  inline bool DeleteTagFromBucket(const size_t i, const uint32_t tag) {
    if (!delete_supported) {
      return false;
    }
    // no delete support
  }

  inline bool InsertTagToBucket(word_t* __restrict filter_data,
                                const size_t i, const uint32_t tag,
                                const bool kickout, uint32_t &oldtag) const {
    for (size_t j = 0; j < kTagsPerBucket; j++) {
      auto t = ReadTag(filter_data, i, j);
      if (t == tag) {
        return true;
      }
      if (t == 0) {
        WriteTag(filter_data, i, j, tag);
        return true;
      }
    }
    if (kickout) {
      size_t r = rand() % kTagsPerBucket; // horribly slow, and causes thread synchronization
      oldtag = ReadTag(filter_data, i, r);
      WriteTag(filter_data, i, r, tag);
    }
    return false;
  }

  inline size_t NumTagsInBucket(const word_t* __restrict filter_data,
                                const size_t i) const {
    size_t num = 0;
    for (size_t j = 0; j < kTagsPerBucket; j++) {
      if (ReadTag(filter_data, i, j) != 0) {
        num++;
      }
    }
    return num;
  }

  inline size_t NumOccupiedEntries(const word_t* __restrict filter_data) const {
    std::size_t num = 0;
    for (size_t bucket_idx = 0; bucket_idx < num_buckets_; bucket_idx++) {
      for (size_t tag_idx = 0; tag_idx < kTagsPerBucket; tag_idx++) {
        if (ReadTag(filter_data, bucket_idx, tag_idx) != 0) {
          num++;
        }
      }
    }
    return num;
  }

};

}  // namespace cuckoofilter
