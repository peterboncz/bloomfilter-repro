#pragma once

#include <cmath>

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>
#include <dtl/bloomfilter/dynamic/blocked_bloomfilter.hpp>

#include "api.hpp"


template<>
struct FilterAPI<dtl::bloomfilter_dynamic::blocked_bloomfilter> {

  using filter_t = dtl::bloomfilter_dynamic::blocked_bloomfilter;

  static constexpr u1 supports_concurrent_inserts = false;

  template <typename ...Params>
  static filter_t construct(Params&&... params) {
    return filter_t(std::forward<Params>(params)...);
  }


  __forceinline__
  static void
  add(uint64_t key, filter_t* filter) {
    filter->insert(key);
  }


  __forceinline__
  static bool
  contain(uint64_t key, const filter_t* filter) {
    return filter->contains(key);
  }


  template<typename input_it, typename consumer_fn>
  static void
  contain(input_it begin, input_it end, const filter_t* filter, consumer_fn consumer) {
    dtl::batch_wise(begin, end, [&](const auto batch_begin, const auto batch_end) {
      std::size_t match_count = 0;
      uint32_t match_pos[dtl::BATCH_SIZE];
      match_count += filter->batch_contains(&batch_begin[0], batch_end - batch_begin, match_pos, 0);
      consumer(match_pos, match_pos + match_count);
    });
  }


  static std::string
  name(const filter_t* filter) {
    return filter->filter_logic.info();
  }


  static std::string
  info(const filter_t* /*filter*/) {
    return "n/a";
  }


  static std::size_t
  size_in_bytes(const filter_t* filter) {
    return filter->filter_data.size() * sizeof(typename filter_t::word_t);
  }

};
