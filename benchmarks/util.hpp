#pragma once

#include <algorithm>
#include <bitset>
#include <functional>
#include <random>
#include <unordered_set>

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>


//===----------------------------------------------------------------------===//
using vector_t = std::vector<uint32_t>;
using numa_vector_t = std::vector<uint32_t, dtl::mem::numa_allocator<uint32_t>>;


// The (static) unrolling factor
constexpr size_t UNROLL_FACTOR = 16;
//===----------------------------------------------------------------------===//






//===----------------------------------------------------------------------===//
namespace internal {


static void
gen_random_data(vector_t& data,
                const std::size_t element_cnt,
                const bool unique_elements,
                const std::function<uint32_t()>& rnd) {
  data.clear();
  data.reserve(element_cnt);

  if (unique_elements) { // Generate unique elements.

    if (element_cnt > (1ull << 32)) {
      std::cerr << "Cannot create more than 2^32 unique integers." << std::endl;
      std::exit(1);
    }

    if (element_cnt == (1ull << 32)) {

      // Entire integer domain.
      for (std::size_t i = 0; i < element_cnt; i++) {
        data.push_back(static_cast<uint32_t>(i));
      }
      std::random_device rnd_device;
      std::shuffle(data.begin(), data.end(), rnd_device);

    }
    else {

      auto is_in_set = new std::bitset<1ull<<32>;
      std::size_t c = 0;
      while (c < element_cnt) {
        auto val = rnd();
        if (!(*is_in_set)[val]) {
          data.push_back(val);
          (*is_in_set)[val] = true;
          c++;
        }
      }
      delete is_in_set;

    }
  }
  else {  // Generate non-unique elements.
    for (std::size_t i = 0; i < element_cnt; i++) {
      data.push_back(rnd());
    }
  }
}


static void
gen_random_data_64(std::vector<uint64_t>& data,
                   const std::size_t element_cnt,
                   const bool unique_elements,
                   const std::function<uint64_t()>& rnd) {
  data.clear();
  data.reserve(element_cnt);

  if (unique_elements) { // Generate unique elements.

    if (element_cnt > (1ull << 32)) {
      std::cerr << "Cannot create more than 2^32 unique integers." << std::endl;
      std::exit(1);
    }

    std::unordered_set<uint64_t> set;
    std::size_t c = 0;
    while (c < element_cnt) {
      auto val = rnd();
      if (set.count(val) == 0) {
        data.push_back(val);
        set.insert(val);
        c++;
      }
    }
  }
  else {  // Generate non-unique elements.
    for (std::size_t i = 0; i < element_cnt; i++) {
      data.push_back(rnd());
    }
  }
}

} // namespace internal


enum rnd_engine_t {
  RANDOM_DEVICE,
  MERSENNE_TWISTER,
};


static void
gen_data(std::vector<uint32_t>& data,
         const std::size_t element_cnt,
         const rnd_engine_t rnd_engine,
         const bool unique) {

  std::random_device rnd_device;

  switch (rnd_engine) {
    case RANDOM_DEVICE: {
      auto gen_rand = [&rnd_device]() {
        return static_cast<uint32_t>(rnd_device());
      };
      internal::gen_random_data(data, element_cnt, unique, gen_rand);
      break;

    }
    case MERSENNE_TWISTER:  {
      auto gen_rand = [&rnd_device]() {
        std::mt19937 gen(rnd_device());
        std::uniform_int_distribution<uint32_t> dis;
        return dis(gen);
      };
      internal::gen_random_data(data, element_cnt, unique, gen_rand);
      break;
    }
  }
}

static void
gen_data(std::vector<uint64_t>& data,
         const std::size_t element_cnt,
         const rnd_engine_t rnd_engine,
         const bool unique) {

  std::random_device rnd_device;

  switch (rnd_engine) {
    case RANDOM_DEVICE: {
      auto gen_rand = [&rnd_device]() {
        return rnd_device() + (static_cast<::std::uint64_t>(rnd_device()) << 32);
      };
      internal::gen_random_data_64(data, element_cnt, unique, gen_rand);
      break;

    }
    case MERSENNE_TWISTER:  {
      std::mt19937_64 gen(rnd_device());
      std::uniform_int_distribution<uint64_t> dis;
      auto gen_rand = [&]() {
        return dis(gen);
      };
      internal::gen_random_data_64(data, element_cnt, unique, gen_rand);
      break;
    }
  }
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
struct data_generator_64 {

  const rnd_engine_t rnd_engine;
  u1 unique_elements;
  std::unordered_set<uint64_t> set;
  std::function<uint64_t()> rnd_fn;

  data_generator_64(const rnd_engine_t rnd_engine, u1 unique_elements)
      : rnd_engine(rnd_engine),
        unique_elements(unique_elements) {

    std::random_device rnd_device;
    std::mt19937_64 gen(rnd_device());
    std::uniform_int_distribution<uint64_t> dis;

    std::function<uint64_t()> rnd = [&rnd_device]() {
      return rnd_device() + (static_cast<::std::uint64_t>(rnd_device()) << 32);
    };

    std::function<uint64_t()> mt = [&]() {
      return dis(gen);
    };

    switch (rnd_engine) {
      case RANDOM_DEVICE:
        rnd_fn = rnd;
        break;
      case MERSENNE_TWISTER:
        rnd_fn = mt;
        break;
    }

  }


  uint64_t
  next() {
    if (unique_elements) { // Generate unique elements.
      while (true) {
        auto val = rnd_fn();
        if (set.count(val) == 0) {
          return val;
        }
      }
    }
    else {  // Generate non-unique elements.
      return rnd_fn();
    }
  }

};
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
struct data_generator_32 {

  u1 unique_elements;
  std::unordered_set<uint32_t> set;

  explicit
  data_generator_32(u1 unique_elements)
    :unique_elements(unique_elements) {};

  virtual uint32_t
  next() = 0;

};

struct data_generator_32_rnd : data_generator_32 {

  std::random_device rnd_device;

  explicit
  data_generator_32_rnd(u1 unique_elements)
      : data_generator_32(unique_elements) {}


  uint32_t
  next() override {
    if (unique_elements) { // Generate unique elements.
      while (true) {
        auto val = rnd_device();
        if (set.count(val) == 0) {
          return val;
        }
      }
    }
    else {  // Generate non-unique elements.
      return rnd_device();
    }
  }

};


struct data_generator_32_mt : data_generator_32 {

  std::random_device rnd_device;
  std::mt19937 gen;
  std::uniform_int_distribution<uint32_t> dis;

  explicit
  data_generator_32_mt(u1 unique_elements)
      : data_generator_32(unique_elements), gen(rnd_device()) { }


  uint32_t
  next() override {
    if (unique_elements) { // Generate unique elements.
      while (true) {
        auto val = dis(gen);
        if (set.count(val) == 0) {
          return val;
        }
      }
    }
    else {  // Generate non-unique elements.
      return dis(gen);
    }
  }

};
//===----------------------------------------------------------------------===//
