#pragma once

#include <string>

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>
#include <dtl/math.hpp>

#include <dtl/filter/bbf_32.hpp>
#include <dtl/filter/bbf_64.hpp>
#include <dtl/filter/zbbf_32.hpp>
#include <dtl/filter/zbbf_64.hpp>

#include "api.hpp"



template<>
struct FilterAPI<dtl::bbf_32> {

  using filter_t = dtl::bbf_32;
  using word_t = typename filter_t::word_t;

  static constexpr u1 supports_concurrent_inserts = true;

  static filter_t construct(const std::size_t bitlength, const uint32_t k, const uint32_t w, const uint32_t s) {
    filter_t bf(bitlength, k, w, s);
    return std::move(bf);
  }


  __forceinline__
  static void
  add(uint64_t key, filter_t* filter, word_t* __restrict filter_data) {
    filter->insert(filter_data, key);
  }


  __forceinline__
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
    return filter->name();
  }


  static std::string
  info(const filter_t* /*filter*/) {
    return "n/a";
  }


  static std::size_t
  size_in_bytes(const filter_t* filter) {
    return filter->size_in_bytes();
  }


  static std::size_t
  size(const filter_t* filter) {
    return filter->size();
  }

};
//===----------------------------------------------------------------------===//


template<>
struct FilterAPI<dtl::bbf_64> {

  using filter_t = dtl::bbf_64;
  using word_t = typename filter_t::word_t;

  static constexpr u1 supports_concurrent_inserts = true;

  static filter_t construct(const std::size_t bitlength, const uint32_t k, const uint32_t w, const uint32_t s) {
    filter_t bf(bitlength, k, w, s);
    return std::move(bf);
  }


  __forceinline__
  static void
  add(uint64_t key, filter_t* filter, word_t* __restrict filter_data) {
    filter->insert(filter_data, key);
  }


  __forceinline__
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
    return filter->name();
  }


  static std::string
  info(const filter_t* /*filter*/) {
    return "n/a";
  }


  static std::size_t
  size_in_bytes(const filter_t* filter) {
    return filter->size_in_bytes();
  }


  static std::size_t
  size(const filter_t* filter) {
    return filter->size();
  }

};
//===----------------------------------------------------------------------===//


template<>
struct FilterAPI<dtl::zbbf_32> {

  using filter_t = dtl::zbbf_32;
  using word_t = typename filter_t::word_t;

  static constexpr u1 supports_concurrent_inserts = true;

  static filter_t construct(const std::size_t bitlength, const uint32_t k, const uint32_t w, const uint32_t s) {
    filter_t bf(bitlength, k, w, s);
    return std::move(bf);
  }


  __forceinline__
  static void
  add(uint64_t key, filter_t* filter, word_t* __restrict filter_data) {
    filter->insert(filter_data, key);
  }


  __forceinline__
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
    return filter->name();
  }


  static std::string
  info(const filter_t* /*filter*/) {
    return "n/a";
  }


  static std::size_t
  size_in_bytes(const filter_t* filter) {
    return filter->size_in_bytes();
  }


  static std::size_t
  size(const filter_t* filter) {
    return filter->size();
  }

};
//===----------------------------------------------------------------------===//
template<>
struct FilterAPI<dtl::zbbf_64> {

  using filter_t = dtl::zbbf_64;
  using word_t = typename filter_t::word_t;

  static constexpr u1 supports_concurrent_inserts = true;

  static filter_t construct(const std::size_t bitlength, const uint32_t k, const uint32_t w, const uint32_t s) {
    filter_t bf(bitlength, k, w, s);
    return std::move(bf);
  }


  __forceinline__
  static void
  add(uint64_t key, filter_t* filter, word_t* __restrict filter_data) {
    filter->insert(filter_data, key);
  }


  __forceinline__
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
    return filter->name();
  }


  static std::string
  info(const filter_t* /*filter*/) {
    return "n/a";
  }


  static std::size_t
  size_in_bytes(const filter_t* filter) {
    return filter->size_in_bytes();
  }


  static std::size_t
  size(const filter_t* filter) {
    return filter->size();
  }

};
//===----------------------------------------------------------------------===//

