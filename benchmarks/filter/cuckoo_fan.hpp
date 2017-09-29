#pragma once

#include "api.hpp"
#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>
#include "cuckoofilter.h"


template<
    typename ItemType,
    size_t bits_per_item,
    size_t associativity,
    template<size_t, size_t> class TableType,
    typename HashFamily
>
struct FilterAPI<cuckoofilter::CuckooFilter<ItemType, bits_per_item, associativity, TableType, HashFamily>> {

  using filter_t = cuckoofilter::CuckooFilter<ItemType, bits_per_item, associativity, TableType, HashFamily>;
  using word_t = typename filter_t::word_t;
  using table_t = TableType<bits_per_item, associativity>;

  static constexpr u1 supports_concurrent_inserts = false;

  template <typename ...Params>
  static filter_t construct(Params&&... params) {
    return filter_t(std::forward<Params>(params)...);
  }


  __forceinline__
  static void
  add(ItemType key, filter_t* table, word_t* __restrict filter_data) {
    if (cuckoofilter::OK != table->Add(filter_data, key)) {
      throw std::logic_error("The filter is too small to hold all of the elements");
    }
  }


  __forceinline__
  static bool
  contain(ItemType key, const filter_t* table, const word_t* __restrict filter_data) {
    return (cuckoofilter::OK == table->Contain(filter_data, key));
  }


  template<typename input_it, typename consumer_fn>
  __unroll_loops__
  static void
  contain(input_it begin, input_it end,
          const filter_t* table, const word_t* __restrict filter_data,
          consumer_fn consumer) {
    dtl::batch_wise(begin, end, [&](const auto batch_begin, const auto batch_end) {
      uint32_t match_pos[dtl::BATCH_SIZE];
      uint32_t* match_writer = match_pos;

      const auto input_size = batch_end - batch_begin;
      const auto mini_batch_cnt = input_size / UNROLL_FACTOR;
      const auto remaining_element_cnt = input_size % UNROLL_FACTOR;
      for (auto mb = 0ull; mb < mini_batch_cnt; mb++) {
        for (auto i = batch_begin + (mb * UNROLL_FACTOR);
             i < (batch_begin + ((mb + 1) * UNROLL_FACTOR));
             i++) {
          auto is_contained = (cuckoofilter::OK == table->Contain(filter_data, *i));
          auto pos = i - batch_begin;
          *match_writer = pos;
          match_writer += is_contained;
        }
      }

      for (auto i = batch_begin + (mini_batch_cnt * UNROLL_FACTOR);
           i < batch_begin + (mini_batch_cnt * UNROLL_FACTOR) + remaining_element_cnt;
           i++) {
        auto is_contained = (cuckoofilter::OK == table->Contain(filter_data, *i));
        auto pos = i - batch_begin;
        *match_writer = pos;
        match_writer += is_contained;
      }
      consumer(match_pos, match_writer);
    });
  }


  static std::string
  name(const filter_t* table) {
    return "{\"name\":\"cuckoo_fan\",\"size\":" + std::to_string(size_in_bytes(table))
        + ",\"tag_bits\":" + std::to_string(bits_per_item)
        + ",\"associativity\":" + std::to_string(TableType<bits_per_item, associativity>::kTagsPerBucket)
        + ",\"delete_support\":" + std::to_string(TableType<bits_per_item, associativity>::delete_supported)
        + ",\"count_support\":" + std::to_string(TableType<bits_per_item, associativity>::counting_supported)
        + ",\"addr\":" + "\"pow2\""
        + "}";
  }


  static std::string
  info(const filter_t* table) {
    return "n/a"; //table->Info();
  }


  static std::size_t
  size_in_bytes(const filter_t* table) {
    return table->SizeInBytes();
  }

};
