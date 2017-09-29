#pragma once

#include "api.hpp"
#include "simd-block.h"

template<>
struct FilterAPI<SimdBlockFilter<>> {

  using filter_t = SimdBlockFilter<>;
  using word_t = typename filter_t::word_t;

  static constexpr u1 supports_concurrent_inserts = false;

  static filter_t construct(const std::size_t bitlength) {
    return filter_t(dtl::log_2(dtl::next_power_of_two(bitlength) / 8));
  }


  __forceinline__
  static void
  add(uint64_t key, filter_t* filter, word_t* __restrict filter_data) {
    filter->Add(filter_data, key);
  }


  __forceinline__
  static bool
  contain(uint64_t key, const filter_t* filter, const word_t* __restrict filter_data) {
    return filter->Find(filter_data, key);
  }


  template<typename input_it, typename consumer_fn>
  __unroll_loops__
  static void
  contain(input_it begin, input_it end,
          const filter_t* filter, const word_t* __restrict filter_data,
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
          auto is_contained = (*filter).Find(filter_data, *i);
          auto pos = i - batch_begin;
          *match_writer = pos;
          match_writer += is_contained;
        }
      }

      for (auto i = batch_begin + (mini_batch_cnt * UNROLL_FACTOR);
                i < batch_begin + (mini_batch_cnt * UNROLL_FACTOR) + remaining_element_cnt;
                i++) {
        auto is_contained = (*filter).Find(filter_data, *i);
        auto pos = i - batch_begin;
        *match_writer = pos;
        match_writer += is_contained;
      }
      consumer(match_pos, match_writer);
    });
  }


  static std::string
  name(const filter_t* filter) {
    return "{\"name\":\"blocked_bloom_impala\", \"size\":" + std::to_string(size_in_bytes(filter)) + "}";
  }


  static std::string
  info(const filter_t* /*filter*/) {
    return "n/a";
  }


  static std::size_t
  size_in_bytes(const filter_t* filter) {
    return filter->SizeInBytes();
  }


  static std::size_t
  size(const filter_t* filter) {
    return filter->size();
  }

};
