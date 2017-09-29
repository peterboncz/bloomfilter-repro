#pragma once

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>

#include <dtl/filter/bloomfilter/bloomfilter_logic.hpp>

#include "api.hpp"


template<dtl::block_addressing AddrMode>
struct FilterAPI<dtl::bloomfilter_logic<uint32_t, AddrMode>> {

  using filter_t = dtl::bloomfilter_logic<uint32_t, AddrMode>;
  using word_t = typename filter_t::word_t;

  static constexpr u1 supports_concurrent_inserts = false;

  template <typename ...Params>
  static filter_t construct(Params&&... params) {
    return filter_t(std::forward<Params>(params)...);
  }


  static void
  add(uint64_t key, filter_t* filter, word_t* __restrict filter_data) {
    filter->insert(filter_data, key);
  }


  static bool
  contain(uint64_t key, const filter_t* filter, const word_t* __restrict filter_data) {
    return filter->contains(filter_data, key);
  }


  template<typename input_it, typename consumer_fn>
  static void
  contain(input_it begin, input_it end,
          const filter_t* filter, const word_t* __restrict filter_data,
          consumer_fn consumer) {
    dtl::batch_wise(begin, end, [&](const auto batch_begin, const auto batch_end) {
      std::size_t match_count = 0;
      uint32_t match_pos[dtl::BATCH_SIZE];
      match_count += filter->batch_contains(filter_data, &batch_begin[0], batch_end - batch_begin, match_pos, 0);
      consumer(match_pos, match_pos + match_count);
    });
  }


  static std::string
  name(const filter_t* filter) {
    return "{\"name\":\"bloom\", \"size\":" + std::to_string(size_in_bytes(filter))
        + ",\"k\":" + std::to_string(filter->k)
        + ",\"addr\":" + (AddrMode == dtl::block_addressing::POWER_OF_TWO ? "\"pow2\"" : "\"magic\"")
        + "}";
  }


  static std::string
  info(const filter_t* /*filter*/) {
    return "n/a";
  }


  static std::size_t
  size_in_bytes(const filter_t* filter) {
    return filter->size() * sizeof(typename filter_t::word_t);
  }

};
