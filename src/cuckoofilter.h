#pragma once

#include <assert.h>
#include <bitset>

#include "debug.h"
#include "hashutil.h"
#include "packedtable.h"
#include "printutil.h"
#include "singletable.h"
#include "bettertable.h"
#include "bettertable_counting.h"

#include <dtl/dtl.hpp>
#include <dtl/math.hpp>
#include <dtl/filter/blocked_bloomfilter/block_addressing_logic.hpp>

namespace cuckoofilter {


//===----------------------------------------------------------------------===//
// Status returned by a cuckoo filter operation.
enum status {
  OK = 0,
  NOT_FOUND = 1,
  NOT_ENOUGH_SPACE = 2,
  NOT_SUPPORTED = 3,
};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Maximum number of cuckoo kicks before claiming failure.
const std::size_t kMaxCuckooCount = 500;
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// A cuckoo filter class exposes a Bloomier filter interface,
// providing methods of Add, Delete, contain. It takes three
// template parameters:
//   key_t:  the type of item you want to insert
//   bits_per_item: how many bits each item is hashed into
//   table_t: the storage of table, SingleTable by default, and
// PackedTable to enable semi-sorting
//===----------------------------------------------------------------------===//
template <typename key_t,
          std::size_t bits_per_tag,
          std::size_t tags_per_bucket,
          template <std::size_t, std::size_t> class table_t,
          typename hash_family = TwoIndependentMultiplyShift
>
class CuckooFilter {

public:
  using word_t = typename table_t<bits_per_tag, tags_per_bucket>::word_t;
  using key_type = key_t;

  // An overflow entry for a single item (used when the filter became full)
  typedef struct {
    size_t index;
    uint32_t tag;
    bool used;
  } victim_cache_t;


  // Table logic
  const table_t<bits_per_tag, tags_per_bucket> table;

  // The victim cache is stored at the very end of the filter data.
  const std::size_t victim_cache_offset;

  // The hasher.
  const hash_family hasher;

  static constexpr std::size_t
      bucket_bitlength = table_t<bits_per_tag, tags_per_bucket>::kBytesPerBucket * 8;

  //===----------------------------------------------------------------------===//
  explicit
  CuckooFilter(const std::size_t desired_bitlength)
      : table(dtl::next_power_of_two(desired_bitlength / 8 / table_t<bits_per_tag, tags_per_bucket>::kBytesPerBucket)),
        victim_cache_offset(table.SizeInBytes() / sizeof(word_t)),
        hasher() {
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Maps a 32-bit hash value to a bucket index.
  inline size_t
  IndexHash(uint32_t hash_val) const {
    // table->num_buckets is always a power of two, so modulo can be replaced
    // with
    // bitwise-and:
    return hash_val & (table.NumBuckets() - 1);
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Derives a (valid) tag from the given 32 bit hash value.
  inline uint32_t
  TagHash(uint32_t hash_value) const {
    uint32_t tag;
    tag = hash_value & static_cast<uint32_t>((1ull << bits_per_tag) - 1);
    tag += (tag == 0);
    return tag;
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Hash the key and derive the tag and the bucket index.
  inline void
  GenerateIndexTagHash(const key_t& key, std::size_t* index, uint32_t* tag) const {
    const uint64_t hash_value = hasher(key);
    *index = IndexHash(static_cast<uint32_t>(hash_value >> 32));
    *tag = TagHash(static_cast<uint32_t>(hash_value));
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  /// Compute the alternative bucket index
  inline std::size_t
  AltIndex(const std::size_t bucket_idx, const uint32_t tag) const {
    // NOTE(binfan): originally we use:
    // bucket_idx ^ HashUtil::BobHash((const void*) (&tag), 4)) & table->INDEXMASK;
    // now doing a quick-n-dirty way:
    // 0x5bd1e995 is the hash constant from MurmurHash2
    return IndexHash((uint32_t)(bucket_idx ^ (tag * 0x5bd1e995)));
  }
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  status
  AddImpl(word_t* __restrict filter_data, const std::size_t i, const uint32_t tag) const {
    std::size_t curindex = i;
    uint32_t curtag = tag;
    uint32_t oldtag;

    for (uint32_t count = 0; count < kMaxCuckooCount; count++) {
      bool kickout = count > 0;
      oldtag = 0;
      if (table.InsertTagToBucket(filter_data, curindex, curtag, kickout, oldtag)) {
        return OK;
      }
      if (kickout) {
        curtag = oldtag;
      }
      curindex = AltIndex(curindex, curtag);
    }

    victim_cache_t& victim = *reinterpret_cast<victim_cache_t*>(&filter_data[victim_cache_offset]);
    victim.index = curindex;
    victim.tag = curtag;
    victim.used = true;
    return OK;
  }
  //===----------------------------------------------------------------------===//


public:

  //===----------------------------------------------------------------------===//
  /// Add an item to the filter.
  status
  Add(word_t* __restrict filter_data, const key_t &key) {

    std::size_t i;
    uint32_t tag;
    const victim_cache_t& victim = *reinterpret_cast<const victim_cache_t*>(&filter_data[victim_cache_offset]);
    if (victim.used) {
      return NOT_ENOUGH_SPACE;
    }

    GenerateIndexTagHash(key, &i, &tag);
    return AddImpl(filter_data, i, tag);
  };
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Report if the item is inserted, with false positive rate.
  status
  Contain(const word_t* __restrict filter_data, const key_t &key) const {
    bool found = false;
    size_t i1, i2;
    uint32_t tag;

    GenerateIndexTagHash(key, &i1, &tag);
    i2 = AltIndex(i1, tag);

    assert(i1 == AltIndex(i2, tag));

    const victim_cache_t& victim = *reinterpret_cast<const victim_cache_t*>(&filter_data[victim_cache_offset]);
    found = victim.used && (tag == victim.tag) &&
            (i1 == victim.index || i2 == victim.index);

    if (found || table.FindTagInBuckets(filter_data, i1, i2, tag)) {
      return OK;
    }
    else {
      return NOT_FOUND;
    }
  }
  //===----------------------------------------------------------------------===//


//  // Delete an key from the filter
//  status Delete(const key_t &item);

  /* methods for providing stats  */
  // summary information
//  std::string Info() const;

//  // number of current inserted items;
//  size_t Size() const { return num_items_; }

  // size of the filter in bytes.
  std::size_t SizeInBytes() const { return table.SizeInBytes(); }

  // Returns the total size of the filter data (including victim cache).
  // Used to allocate memory. E.g., std::vector<word_t> filter_data(size(),0);
  std::size_t size() const {
    return ((table.SizeInBytes() + sizeof(victim_cache_t)) + (sizeof(word_t) - 1)) / sizeof(word_t);
  }

  std::size_t countOccupiedSlots(const word_t* __restrict filter_data) const {
    const victim_cache_t& victim = *reinterpret_cast<const victim_cache_t*>(&filter_data[victim_cache_offset]);
    return table.NumOccupiedEntries(filter_data) + victim.used;
  }

  std::vector<std::size_t>
  slotOccupationHistogram(const word_t* __restrict filter_data) const {
    std::vector<std::size_t> histo(slotCountPerBucket() + 1, 0);
    for (std::size_t bucket_idx = 0; bucket_idx < bucketCount(); bucket_idx++) {
      histo[table.NumTagsInBucket(filter_data, bucket_idx)]++;
    }
    return histo;
  }

  std::size_t bucketCount() const {
    return table.NumBuckets();
  }

  std::size_t slotCount() const {
    return table.SizeInTags() + 1 /* victim cache */;
  }

  std::size_t slotCountPerBucket() const {
    return table.kTagsPerBucket;
  }

  std::size_t tagSizeBits() const {
    return bits_per_tag;
  }

//  __attribute_noinline__
//  $u64 batch_contains(const __restrict word_t* /*filter_ data*/,
//                      const key_t* /*keys*/, u32 /*key_cnt*/,
//                      $u32* /*match_positions*/, u32 /*match_offset*/) {
//
//  }

};
//===----------------------------------------------------------------------===//



//template <typename key_t, size_t bits_per_item,
//          template <size_t> class table_t, typename hash_family>
//status CuckooFilter<key_t, bits_per_item, table_t, hash_family>::Delete(
//    word_t* __restrict filter_data,
//    const key_t &key) {
//  size_t i1, i2;
//  uint32_t tag;
//
//  generate_index_tag_hash(key, &i1, &tag);
//  i2 = AltIndex(i1, tag);
//
//  if (table->DeleteTagFromBucket(i1, tag)) {
//    num_items_--;
//    goto TryEliminateVictim;
//  } else if (table->DeleteTagFromBucket(i2, tag)) {
//    num_items_--;
//    goto TryEliminateVictim;
//  } else if (victim_.used && tag == victim_.tag &&
//             (i1 == victim_.index || i2 == victim_.index)) {
//    // num_items_--;
//    victim_.used = false;
//    return OK;
//  } else {
//    return NOT_FOUND;
//  }
//TryEliminateVictim:
//  if (victim_.used) {
//    victim_.used = false;
//    size_t i = victim_.index;
//    uint32_t tag = victim_.tag;
//    AddImpl(i, tag);
//  }
//  return OK;
//}

}  // namespace cuckoofilter
