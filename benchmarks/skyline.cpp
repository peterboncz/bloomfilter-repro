#include <climits>
#include <fstream>
#include <iomanip>
#include <map>
#include <numa.h>
#include <numaif.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "random.h"
#include "timing.h"

#include <dtl/dtl.hpp>
#include <dtl/batchwise.hpp>
#include <dtl/barrier.hpp>
#include <dtl/env.hpp>
#include <dtl/filter/blocked_bloomfilter/fpr.hpp>
#include <dtl/mem.hpp>
#include <dtl/simd.hpp>
#include <dtl/thread.hpp>

#include <boost/algorithm/string.hpp>

#include "filters.hpp"

//===----------------------------------------------------------------------===//
// The benchmark runner.
//===----------------------------------------------------------------------===//

// Read the CPU affinity for the process.
static const auto cpu_mask = dtl::this_thread::get_cpu_affinity();

const std::size_t SAMPLE_SIZE = 1ull << 26; // 256 MiB

struct Statistics {
  uint64_t match_count;
  uint64_t avg_lookup_duration;
  uint64_t avg_tsc;
};


struct config_t {
  std::string filter_name;
  /// Filter size in bits
  uint64_t m = 0;
  /// Number of elements
//  uint64_t n = 0;
  std::set<uint64_t> n {}; // multiple n's in ascending order
  /// Bloom: number of hash functions
  uint64_t k = 0;
  /// Blocked Bloom: number of words per block
  uint64_t word_cnt_per_block = 0;
  /// Blocked Bloom: number of sectors per block
  uint64_t sector_cnt_per_block = 0;
  /// Blocked Cuckoo: The block size in bytes
  uint32_t block_size_bytes = 0;
  /// Cuckoo: Number of bits per tag (aka key signature)
  uint32_t tag_bits = 0;
  /// Cuckoo: Number of slots per bucket
  uint32_t associativity = 0;
//  /// The benchmark mode (precision | performance | full)
//  benchmark_mode_t benchmark_mode;
  /// Blocked Bloom: number of zones per block
  uint64_t zone_cnt_per_block = 1;
};

// Precision and performance benchmark can be run separately.
static const bool bench_precision = dtl::env<$u64>::get("BENCH_PRECISION", 1) > 0;
static const bool bench_performance = dtl::env<$u64>::get("BENCH_PERFORMANCE", 1) > 0;

// Power of two - addressing (if disabled, blocked bloom and blocked cuckoo will only use sizes with are not powers of two)
static const bool pow2_addressing = dtl::env<$u64>::get("POW2_ADDR", 1) > 0; // enabled by default
// Magic-addressing (if disabled, blocked bloom and blocked cuckoo will only use pow2 sizes)
static const bool magic_addressing = dtl::env<$u64>::get("MAGIC_ADDR", 1) > 0; // enabled by default

// Enforce use of non-SIMD code
static const bool force_scalar_code = dtl::env<$u64>::get("FORCE_SCALAR_CODE", 0) > 0;

// Enable validation code
static const bool validation = dtl::env<$u64>::get("VALIDATION", 0) > 0;

static std::vector<$u32> thread_id_to_node_id_map(4096, 0);


//===----------------------------------------------------------------------===//
// Parallel batch lookups.
//===----------------------------------------------------------------------===//
template<typename FilterType>
struct ParallelFilterAPI {

  using filter_t = FilterType;
  using word_t = typename filter_t::word_t;

  using filter_data_vector_t = std::vector<word_t, dtl::mem::numa_allocator<word_t>>;

  template<typename input_it>
  static Statistics
  contain(input_it begin, input_it end,
          const filter_t* filter,
          const std::vector<filter_data_vector_t>& filter_replicas,
          const uint64_t thread_cnt) {
    Statistics results;
    dtl::busy_barrier_one_shot barrier(thread_cnt);

    std::vector<uint64_t> lookup_duration_time(thread_cnt);

    std::vector<uint64_t> lookup_start_time(thread_cnt);
    std::vector<uint64_t> lookup_end_time(thread_cnt);

    std::vector<uint64_t> match_counts(thread_cnt);

    std::vector<uint64_t> tsc_begin(thread_cnt);
    std::vector<uint64_t> tsc_end(thread_cnt);

    auto thread_fn = [&](const uint32_t thread_id) {

      const auto node_id = thread_id_to_node_id_map[thread_id];
      const auto filter_data = &filter_replicas[node_id][0];

//      if (node_id != dtl::mem::get_node_of_address(filter_data)) {
//        std::cerr << "Allocation error." << std::endl;
//      }

      // Each thread starts reading from a different offset
      // to force accesses to different memory locations
      // and to stress the memory bus.
      const auto start_offset = (thread_id * (std::distance(begin, end) / thread_cnt));

      std::size_t match_count = 0;

      // Wait until all threads have spawned.
      barrier.wait();

      const auto lookup_start_time_local = NowNanos();
      const auto tsc_begin_local = _rdtsc();

      if (!force_scalar_code) {
        // call the batch/simd/data-parallel contains function
        FilterAPI<FilterType>::contain(begin + start_offset, end,
                                       filter, &filter_data[0],
                                       [&](const uint32_t* matches_begin, const uint32_t* matches_end) {
                                         match_count += matches_end - matches_begin;
                                       });
        FilterAPI<FilterType>::contain(begin, begin + start_offset,
                                       filter, &filter_data[0],
                                       [&](const uint32_t* matches_begin, const uint32_t* matches_end) {
                                         match_count += matches_end - matches_begin;
                                       });
      }
      else {
        // sequentially call the (non-inlined) scalar contains function
        uint32_t match_pos[dtl::BATCH_SIZE];
        dtl::batch_wise(begin + start_offset, end, [&](const auto batch_begin, const auto batch_end) {
          uint32_t* match_writer = match_pos;
          for (auto i = batch_begin; i < batch_end; i++) {
            auto is_contained = FilterAPI<FilterType>::contain(*i, filter, &filter_data[0]);
            auto pos = i - begin;
            *match_writer = pos;
            match_writer += is_contained;
            match_count += is_contained;
          }
        });
        dtl::batch_wise(begin, begin + start_offset, [&](const auto batch_begin, const auto batch_end) {
          uint32_t* match_writer = match_pos;
          for (auto i = batch_begin; i < batch_end; i++) {
            auto is_contained = FilterAPI<FilterType>::contain(*i, filter, &filter_data[0]);
            auto pos = i - begin;
            *match_writer = pos;
            match_writer += is_contained;
            match_count += is_contained;
          }
        });

      }

      const auto tsc_end_local = _rdtsc();
      const auto lookup_end_time_local = NowNanos();

      lookup_start_time[thread_id] = lookup_start_time_local;
      lookup_end_time[thread_id] = lookup_end_time_local;
      tsc_begin[thread_id] = tsc_begin_local;
      tsc_end[thread_id] = tsc_end_local;

      match_counts[thread_id] = match_count;
    };

    // Do less repetitions with larger filters, but at least 10.

    std::size_t repetitions = 0;
    const auto begin_time = NowNanos();
    do {
      barrier.reset();
      dtl::run_in_parallel(thread_fn, cpu_mask, thread_cnt);

      // aggregate results
      Statistics stats;
      stats.avg_lookup_duration = 0;
      stats.avg_tsc = 0;
      stats.match_count = 0;
      for (std::size_t i = 0; i < thread_cnt; i++) {
        stats.avg_lookup_duration += lookup_end_time[i] - lookup_start_time[i];
        stats.avg_tsc += tsc_end[i] - tsc_begin[i];
        stats.match_count += match_counts[i];
      }
      stats.avg_lookup_duration /= thread_cnt;
      stats.avg_tsc /= thread_cnt;
      stats.match_count /= thread_cnt;

      if (repetitions == 0) {
        results = stats;
      }
      if (stats.avg_lookup_duration < results.avg_lookup_duration) {
        results = stats;
      }
      repetitions++;
    } while (NowNanos() - begin_time < 20000000);
    return results;
  }

};
//===----------------------------------------------------------------------===//



auto inc = [&](u64 i, u64 step_mode, u64 step_size = 1) {
  if (step_mode == 1) {
    // linear
    return i + step_size;
  }
  else {
    // exponential
    auto step = step_size > 1 ? step_size : 2;
    return i * step;
  }
};


enum class benchmark_mode_t {
  EXCLUSIVE, // the current instance is the only one (is using the CPU exclusively)
  SHARED     // multiple instances are running (and should therefore not spawn new threads)
};


//===----------------------------------------------------------------------===//
// Benchmark runner.
//===----------------------------------------------------------------------===//
template <typename FilterType, typename ItemType = uint32_t, typename ...Params>
void FilterBenchmark(const benchmark_mode_t benchmark_mode,
                     const std::size_t m,
                     const std::set<std::size_t> n_s,
                     const double sel,
                     const std::vector<ItemType, dtl::mem::numa_allocator<uint32_t>>& to_add,
                     const std::vector<ItemType, dtl::mem::numa_allocator<uint32_t>>& to_lookup,
                     Params&&... constructor_params) {

  //===----------------------------------------------------------------------===//
  // Multi-threading
  // repeats the benchmark with different concurrency settings (only in performance-benchmark mode)
  const auto thread_cnt_lo = dtl::env<$u64>::get("THREAD_CNT_LO", 1);
  const auto thread_cnt_hi = dtl::env<$u64>::get("THREAD_CNT_HI", cpu_mask.count());

  // 1 = linear, 2 = exponential
  const auto thread_step_mode = dtl::env<$u64>::get("THREAD_STEP_MODE", 2);
  const auto thread_step = dtl::env<$u64>::get("THREAD_STEP", 1);
  //===----------------------------------------------------------------------===//


  //===----------------------------------------------------------------------===//
  // Test the filter even if not all elements have been inserted. (only relevant for CF)
  // The # of inserted elements must be greater or equal to fill_threshold * n.
  f64 fill_threshold = 1.0; //0.95;
  //===----------------------------------------------------------------------===//

  try {

    //===----------------------------------------------------------------------===//
    // Construct the filter (logic) instance.
    FilterType filter = FilterAPI<FilterType>::construct(std::forward<Params>(constructor_params)...);
    using filter_word_t = typename FilterType::word_t;

    // Allocate memory.
    dtl::mem::allocator_config alloc_config = dtl::mem::allocator_config::local();

    // TODO replicate filter to all NUMA/HBM nodes
    if (dtl::mem::hbm_available()) {
      // For KNL we use the nearest HBM node.
      auto cpu_id = dtl::this_thread::get_cpu_affinity().find_first();
      auto hbm_node_id = dtl::mem::get_nearest_hbm_node(dtl::mem::get_node_of_cpu(cpu_id));
      alloc_config = dtl::mem::allocator_config::on_node(hbm_node_id);
    }

    // Initialize the filter data with zeros.
    dtl::mem::numa_allocator<filter_word_t> allocator(alloc_config);
    std::vector<filter_word_t, dtl::mem::numa_allocator<filter_word_t>>
        filter_data(filter.size() + 32, 0, allocator); // + some additional bytes to work around the cuckoo buffer overrun bug
    //===----------------------------------------------------------------------===//


    //===----------------------------------------------------------------------===//
    // Loop over n values in ascending order
    // to speed up benchmarks as not all elements have the added over and over again.
    std::size_t previous_n = 0;
    for (std::size_t n : n_s) {


      //===----------------------------------------------------------------------===//
      // Initialize the filter -- Construction is done sequentially
      std::size_t added_element_cnt = previous_n;
      try {
        std::for_each(to_add.begin() + previous_n, to_add.begin() + n, [&](auto key) {
          // For cuckoo filter, this will throw an exception if element cannot be added.
          FilterAPI<FilterType>::add(key, &filter, &filter_data[0]);
          added_element_cnt++;
          // Paranoid validation code
          if (validation) {
            if (!FilterAPI<FilterType>::contain(key, &filter, &filter_data[0])) {
              std::cerr << "Failed to insert key " << key << " into filter '" << FilterAPI<FilterType>::name(&filter) << "'!" << std::endl;
              exit(1);
            }
          }
          // --- /validation code
        });
      }
      catch (std::logic_error& e) {
        // Cuckoo filter to small; relax things a bit; otherwise the Cuckoo will disappear from the performance skyline.
        const auto insert_ratio = ((added_element_cnt * 1.0) / n);
        std::stringstream str;
        str << "Failed to add more elements to the filter. insert_ratio=" << insert_ratio
            << " (" << added_element_cnt << " of " << n << ")" << ", filter=" << FilterAPI<FilterType>::name(&filter);

        if (insert_ratio < fill_threshold) {
          // Too few elements, sorry Cuckoo.
          str << " - Abort" << std::endl;
          std::cerr << str.str();
          throw;
        }
        else {
          str << " - Within tolerance. -> Continue" << std::endl;
          std::cerr << str.str();
        }
      }
      //===----------------------------------------------------------------------===//


      //===----------------------------------------------------------------------===//
      // Prepare the keys to lookup.
      auto to_lookup_mixed = to_lookup;
      if (sel > 0.0) {
        if (n == 0) {
          std::cerr << "BAM" << std::endl;
          std::exit(1);
        }
        std::random_device rd;
        std::mt19937 gen(rd());
        for (std::size_t i = 0; i < sel * to_lookup_mixed.size(); i++) {
          to_lookup_mixed[i] = to_add[gen() % n];
        }
        std::shuffle(to_lookup_mixed.begin(), to_lookup_mixed.end(), gen);
      }
      //===----------------------------------------------------------------------===//


      //===----------------------------------------------------------------------===//
      // Validation code for batch lookup.
      if (validation) {
        std::size_t match_count_verify = 0;
        FilterAPI<FilterType>::contain(to_add.begin(), to_add.begin() + added_element_cnt,
                                       &filter, &filter_data[0],
                                       [&](const uint32_t* matches_begin, const uint32_t* matches_end) {
                                         match_count_verify += matches_end - matches_begin;
                                       });

        if (match_count_verify != added_element_cnt) {
          std::stringstream str;
          str << "Validation failed: Expected " << added_element_cnt << " matches, but got " << match_count_verify
              << ". (Filter type: " << FilterAPI<FilterType>::name(&filter) << ")" << std::endl;
          std::cerr << str.str();
        }
      }
      //===----------------------------------------------------------------------===//


      if (benchmark_mode == benchmark_mode_t::SHARED) {
        //===----------------------------------------------------------------------===//
        // Benchmark PRECISION only (no performance metrics are collected).
        //===----------------------------------------------------------------------===//
        // Note: We do not have to parallelize in SHARED mode, as the FilterBenchmark
        //       function might already be running in parallel.
        std::size_t match_count = 0;
        FilterAPI<FilterType>::contain(to_lookup_mixed.begin(), to_lookup_mixed.end(),
                                       &filter, &filter_data[0],
                                       [&](const uint32_t* matches_begin, const uint32_t* matches_end) {
                                         // consumer function
                                         match_count += matches_end - matches_begin;
                                       });
        const auto expected_match_count = SAMPLE_SIZE * sel;
        auto false_positives = match_count - expected_match_count;
        false_positives = (false_positives < 0) ? 0 : false_positives;
        const auto fpr = false_positives / SAMPLE_SIZE;
        std::stringstream str;
        std::string filter_info = FilterAPI<FilterType>::name(&filter);
        boost::replace_all(filter_info, "\"", "\"\""); // escape JSON for CSV output
        str << "\"" << filter_info << "\""
            << "," << m
            << "," << (n > 0 ? (m*1.0/n) : 0)
            << "," << n
            << "," << sel
            << "," << 0   // DC
            << "," << false_positives
            << "," << fpr
            << "," << 0.0 // DC
            << "," << 0.0 // DC
            << "," << 1   // single-threaded
            << "," << (force_scalar_code ? "1" : "0")
            << std::endl;
        std::cout << str.str();
        //===----------------------------------------------------------------------===//
      }
      else { // benchmark_mode == benchmark_mode_t::EXCLUSIVE
        //===----------------------------------------------------------------------===//
        // Benchmark PERFORMANCE
        //===----------------------------------------------------------------------===//

        // Replicate the filter to all (active) NUMA nodes (this is especially important for KNL in SNC mode).
        using filter_data_vector_t = std::vector<filter_word_t, dtl::mem::numa_allocator<filter_word_t>>;
        std::vector<filter_data_vector_t> replicas(dtl::mem::get_node_count());
        const auto numa_nodes = dtl::mem::hbm_available() ? dtl::mem::get_hbm_nodes() : dtl::mem::get_cpu_nodes();

        for (auto node_id : numa_nodes) {
          auto replica_alloc_config = dtl::mem::allocator_config::on_node(node_id);
          dtl::mem::numa_allocator<filter_word_t> replica_allocator(replica_alloc_config);
          filter_data_vector_t replica_filter_data(filter_data.begin(), filter_data.end(), replica_allocator);
          replicas[node_id].swap(replica_filter_data);
        }


        // Vary number of threads.
        for (std::size_t thread_cnt = thread_cnt_lo;
             thread_cnt <= thread_cnt_hi;
             thread_cnt = inc(thread_cnt, thread_step_mode, thread_step)) {

          // Run batch lookup in parallel.
          const auto results = ParallelFilterAPI<FilterType>::contain(to_lookup_mixed.begin(), to_lookup_mixed.end(),
                                                                      &filter, replicas,
                                                                      thread_cnt);

          const double cycles_per_lookup = static_cast<double>(results.avg_tsc) / SAMPLE_SIZE;
          const auto expected_match_count = SAMPLE_SIZE * sel;
          auto false_positives = results.match_count - expected_match_count;
          false_positives = (false_positives < 0) ? 0 : false_positives; // FIXME WT.
          if (false_positives < 0) {
            std::cerr << "The number of false positives may not be negative." << std::endl;
            std::exit(1);
          }
          const auto fpr = false_positives / SAMPLE_SIZE;
          std::stringstream str;
          std::string filter_info = FilterAPI<FilterType>::name(&filter);
          boost::replace_all(filter_info, "\"", "\"\"");
          str << "\"" << filter_info << "\""
              << "," << m
              << "," << (n > 0 ? (m*1.0/n) : 0)
              << "," << n
              << "," << sel
              << "," << 0 // (insert_end_time - insert_start_time) not measured any more
              << "," << false_positives
              << "," << fpr
              << "," << (SAMPLE_SIZE / (results.avg_lookup_duration * 1.0e-9))
              << "," << cycles_per_lookup
              << "," << thread_cnt
              << "," << (force_scalar_code ? "1" : "0")
              << std::endl;
          std::cout << str.str();
          // Validation code (for parallel code path)
          if (validation) {
            auto match_count2 = 0ull;
            FilterAPI<FilterType>::contain(to_lookup_mixed.begin(), to_lookup_mixed.end(),
                                           &filter, &filter_data[0],
                                           [&](const uint32_t* matches_begin, const uint32_t* matches_end) {
                                             // consumer function
                                             match_count2 += matches_end - matches_begin;
                                           });
            if (results.match_count != match_count2) {
              std::stringstream str;
              str << "Validation failed: Parallel code returned " << results.match_count << " matches"
                  << " whereas the sequential code path returned " << match_count2 << " matches." << std::endl;
              std::cout << str.str();
            }
          }
        }
        //===----------------------------------------------------------------------===//
      }


      if (added_element_cnt != n) {
        // Construction failed for the current n.
        std::stringstream str;
        str << "Filter construction failed for the current n = " << n << ". No need to test the remaining n's." << std::endl;
        std::cerr << str.str();
        break; // the loop over n's
      }
      else {
        previous_n = n;
      }

    } // loop over n

  }
  catch (std::exception& err) {
    // Log and go on
    std::stringstream str;
    str << "Aw, snap! Caught: " << err.what() << std::endl;
    std::cout << str.str();
  }
  catch(...) {
    // Log and go on
    std::stringstream str;
    str << "Aw, snap!" << std::endl;
    std::cout << str.str();
  }
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Generate two disjoint key sets.
void
gen_data(std::vector<uint32_t, dtl::mem::numa_allocator<uint32_t>>& to_add,
         std::vector<uint32_t, dtl::mem::numa_allocator<uint32_t>>& to_lookup,
         const std::size_t element_cnt) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis;

  to_add.clear();
  to_add.reserve(element_cnt);
  to_lookup.clear();
  to_lookup.reserve(SAMPLE_SIZE);

  if (element_cnt >= 1ull << 32) {
    for (std::size_t i = 0; i < element_cnt; i++) {
      to_add.push_back(i);
    }
    std::shuffle(to_add.begin(), to_add.end(), gen);
    to_lookup.assign(to_add.begin(), to_add.begin() + SAMPLE_SIZE);
    for (std::size_t i = 0; i < SAMPLE_SIZE; i++) {
      to_add[i] = to_add[i + SAMPLE_SIZE];
    }
    std::shuffle(to_add.begin(), to_add.end(), gen);
  }
  else {
    // generate random sample
    auto is_in_sample = new std::bitset<1ull<<32>;
    std::size_t s = 0;
    while (s < SAMPLE_SIZE) {
      auto val = dis(gen);
      if (!(*is_in_sample)[val]) {
        to_lookup.push_back(val);
        (*is_in_sample)[val] = true;
        s++;
      }
    }
    std::size_t c = 0;
    while (c < element_cnt) {
      auto val = dis(gen);
      if (!(*is_in_sample)[val]) {
        to_add.push_back(val);
        c++;
      }
    }
    delete is_in_sample;
  }

}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// Start the benchmark specified by the given config.
void execute(const benchmark_mode_t mode,
             const config_t& c,
             const double sel,
             const std::vector<uint32_t, dtl::mem::numa_allocator<uint32_t>>& to_add,
             const std::vector<uint32_t, dtl::mem::numa_allocator<uint32_t>>& to_lookup) {

  // Determine whether this is only a simulated (precision) run.
  u1 fast = (dtl::env<$u64>::get("FAST", 0) > 0);


  //===----------------------------------------------------------------------===//
  // Standard Bloom filter
  //===----------------------------------------------------------------------===//
  if (c.filter_name == "std") {
    if (dtl::is_power_of_two(c.m)) {
      using FilterType = dtl::bloomfilter_logic<uint32_t, dtl::block_addressing::POWER_OF_TWO>;
      if (fast) { // ----------------------------------------------- <<< FAST RUN
        try {
          FilterType filter = FilterAPI<FilterType>::construct(c.m, c.k);
          std::string filter_info = FilterAPI<FilterType>::name(&filter);
          boost::replace_all(filter_info, "\"", "\"\""); // escape JSON for CSV output

          for (auto n : c.n) {
            const auto fpr = dtl::bloomfilter::fpr(c.m, n, c.k);
            std::stringstream str;
            str << "\"" << filter_info << "\""
                << "," << c.m
                << "," << (n > 0 ? (c.m*1.0/n) : 0)
                << "," << n
                << "," << sel
                << "," << 0   // DC
                << "," << 0   // false_positives <<<--- DC ?
                << "," << fpr
                << "," << 0.0 // DC
                << "," << 0.0 // DC
                << "," << 1   // single-threaded
                << "," << (force_scalar_code ? "1" : "0")
                << std::endl;
            std::cout << str.str();
          }
        }
        catch (std::exception& err) {
          // Log and go on
          std::stringstream str;
          str << "Aw, snap! Caught: " << err.what() << std::endl;
          std::cout << str.str();
        }
        catch(...) {
          // Log and go on
          std::stringstream str;
          str << "Aw, snap!" << std::endl;
          std::cout << str.str();
        }
      }
      else {
        FilterBenchmark<FilterType>(mode, c.m, c.n, sel, to_add, to_lookup, c.m, c.k);
      }
    }
    else {
      using FilterType = dtl::bloomfilter_logic<uint32_t, dtl::block_addressing::MAGIC>;
      if (fast) { // ---------------------------------------------- <<< FAST RUN
        try {
          FilterType filter = FilterAPI<FilterType>::construct(c.m, c.k);
          std::string filter_info = FilterAPI<FilterType>::name(&filter);
          boost::replace_all(filter_info, "\"", "\"\""); // escape JSON for CSV output

          for (auto n : c.n) {
            const auto fpr = dtl::bloomfilter::fpr(c.m, n, c.k);
            std::stringstream str;
            str << "\"" << filter_info << "\""
                << "," << c.m
                << "," << (n > 0 ? (c.m*1.0/n) : 0)
                << "," << n
                << "," << sel
                << "," << 0   // DC
                << "," << 0   // false_positives <<<--- DC ?
                << "," << fpr
                << "," << 0.0 // DC
                << "," << 0.0 // DC
                << "," << 1   // single-threaded
                << "," << (force_scalar_code ? "1" : "0")
                << std::endl;
            std::cout << str.str();
          }
        }
        catch (std::exception& err) {
          // Log and go on
          std::stringstream str;
          str << "Aw, snap! Caught: " << err.what() << std::endl;
          std::cout << str.str();
        }
        catch(...) {
          // Log and go on
          std::stringstream str;
          str << "Aw, snap!" << std::endl;
          std::cout << str.str();
        }
      }
      else {
        FilterBenchmark<FilterType>(mode, c.m, c.n, sel, to_add, to_lookup, c.m, c.k);
      }
    }
  }

#if defined(__AVX2__)
  //===----------------------------------------------------------------------===//
  // Vectorized Bloom filter (Columbia University)
  //===----------------------------------------------------------------------===//
  if (c.filter_name == "columbia") {
    if (dtl::is_power_of_two(c.m)) {
      using FilterType = columbia::vbf;
      if (fast) { // ---------------------------------------------- <<< FAST RUN
        try {
          FilterType filter = FilterAPI<FilterType>::construct(c.m, c.k);
          std::string filter_info = FilterAPI<FilterType>::name(&filter);
          boost::replace_all(filter_info, "\"", "\"\""); // escape JSON for CSV output

          for (auto n : c.n) {
            const auto fpr = dtl::bloomfilter::fpr(c.m, n, c.k);
            std::stringstream str;
            str << "\"" << filter_info << "\""
                << "," << c.m
                << "," << (n > 0 ? (c.m*1.0/n) : 0)
                << "," << n
                << "," << sel
                << "," << 0   // DC
                << "," << 0   // false_positives <<<--- DC ?
                << "," << fpr
                << "," << 0.0 // DC
                << "," << 0.0 // DC
                << "," << 1   // single-threaded
                << "," << (force_scalar_code ? "1" : "0")
                << std::endl;
            std::cout << str.str();
          }
        }
        catch (std::exception& err) {
          // Log and go on
          std::stringstream str;
          str << "Aw, snap! Caught: " << err.what() << std::endl;
          std::cout << str.str();
        }
        catch(...) {
          // Log and go on
          std::stringstream str;
          str << "Aw, snap!" << std::endl;
          std::cout << str.str();
        }
      }
      else {
        FilterBenchmark<FilterType>(mode, c.m, c.n, sel, to_add, to_lookup, c.m, c.k);
      }
    }
  }
#endif

  //===----------------------------------------------------------------------===//
  // Cuckoo filter
  //===----------------------------------------------------------------------===//
  if (c.filter_name == "cuckoo") {
    using FilterType = dtl::cf;
    if (fast) { // ------------------------------------------------ <<< FAST RUN
      try {
        FilterType filter = FilterAPI<FilterType>::construct(c.m, c.tag_bits, c.associativity);
        std::string filter_info = FilterAPI<FilterType>::name(&filter);
        boost::replace_all(filter_info, "\"", "\"\""); // escape JSON for CSV output

        u64 slot_cnt = c.m / c.tag_bits; // FIXME wrong for 12-bit tags due to padding
        for (auto n : c.n) {
          f64 load_factor = (n * 1.0) / slot_cnt;

          // Simulate construction failures.
          if (c.associativity == 1) { // empirically determined with the original CF implementation
            if (c.tag_bits ==  8 && load_factor > 0.13) continue;
            if (c.tag_bits == 12 && load_factor > 0.46) continue;
            if (load_factor > 0.50) continue;
          }
          if (c.associativity == 2 && load_factor > 0.84) continue; // taken from the paper
          if (c.associativity == 4 && load_factor > 0.95) continue; // taken from the paper
          if (load_factor > 0.98) continue; // taken from the paper

          const auto fpr = dtl::cuckoofilter::fpr(c.associativity, c.tag_bits, load_factor);

//        const auto expected_match_count = SAMPLE_SIZE * sel;
//        const auto match_count =
//        auto false_positives = match_count - expected_match_count;
//        false_positives = (false_positives < 0) ? 0 : false_positives;
          std::stringstream str;
          str << "\"" << filter_info << "\""
              << "," << c.m
              << "," << (n > 0 ? (c.m*1.0/n) : 0)
              << "," << n
              << "," << sel
              << "," << 0   // DC
              << "," << 0   // false_positives <<<--- DC ?
              << "," << fpr
              << "," << 0.0 // DC
              << "," << 0.0 // DC
              << "," << 1   // single-threaded
              << "," << (force_scalar_code ? "1" : "0")
              << std::endl;
          std::cout << str.str();
        } // loop over n's
      }
      catch (std::exception& err) {
        // Log and go on
        std::stringstream str;
        str << "Aw, snap! Caught: " << err.what() << std::endl;
        std::cout << str.str();
      }
      catch(...) {
        // Log and go on
        std::stringstream str;
        str << "Aw, snap!" << std::endl;
        std::cout << str.str();
      }
    }
    else {
      FilterBenchmark<FilterType>(mode, c.m, c.n, sel, to_add, to_lookup, c.m, c.tag_bits, c.associativity);
    }
  }

  //===----------------------------------------------------------------------===//
  // Blocked Cuckoo filter
  //===----------------------------------------------------------------------===//
  if (c.filter_name == "blockedcuckoo") {
    if (fast) { // ------------------------------------------------ <<< FAST RUN
      // TODO
    }
    else {
      using FilterType = dtl::bcf;
      FilterBenchmark<FilterType>(mode, c.m, c.n, sel, to_add, to_lookup, c.m, c.block_size_bytes, c.tag_bits, c.associativity);
    }
  }


#if defined(__AVX2__)
  //===----------------------------------------------------------------------===//
  // Blocked Bloom filter (Impala)
  //===----------------------------------------------------------------------===//
  if (c.filter_name == "impala") {
    if (dtl::is_power_of_two(c.m)) {
      using FilterType = SimdBlockFilter<>;
      if (fast) { // ---------------------------------------------- <<< FAST RUN
        try {
          FilterType filter = FilterAPI<FilterType>::construct(c.m);
          std::string filter_info = FilterAPI<FilterType>::name(&filter);
          boost::replace_all(filter_info, "\"", "\"\""); // escape JSON for CSV output

          for (auto n : c.n) {
            const auto fpr = dtl::bloomfilter::fpr(c.m, n, 8, 256, 32);
            std::stringstream str;
            str << "\"" << filter_info << "\""
                << "," << c.m
                << "," << (n > 0 ? (c.m*1.0/n) : 0)
                << "," << n
                << "," << sel
                << "," << 0   // DC
                << "," << 0   // false_positives <<<--- DC ?
                << "," << fpr
                << "," << 0.0 // DC
                << "," << 0.0 // DC
                << "," << 1   // single-threaded
                << "," << (force_scalar_code ? "1" : "0")
                << std::endl;
            std::cout << str.str();
          }
        }
        catch (std::exception& err) {
          // Log and go on
          std::stringstream str;
          str << "Aw, snap! Caught: " << err.what() << std::endl;
          std::cout << str.str();
        }
        catch(...) {
          // Log and go on
          std::stringstream str;
          str << "Aw, snap!" << std::endl;
          std::cout << str.str();
        }
      }
      else {
        FilterBenchmark<FilterType>(mode, c.m, c.n, sel, to_add, to_lookup, c.m);
      }
    }
  }
#endif // defined(__AVX2__)


  //===----------------------------------------------------------------------===//
  // Blocked Bloom filter (Amsterdam)
  //===----------------------------------------------------------------------===//
  if (c.filter_name == "multiregblocked32" && c.zone_cnt_per_block == 1) {
    using FilterType = dtl::bbf_32;
    if (fast) { // ------------------------------------------------ <<< FAST RUN
      try {
        FilterType filter = FilterAPI<FilterType>::construct(c.m, c.k, c.word_cnt_per_block, c.sector_cnt_per_block);
        std::string filter_info = FilterAPI<FilterType>::name(&filter);
        boost::replace_all(filter_info, "\"", "\"\""); // escape JSON for CSV output

        for (auto n : c.n) {
          const auto B = c.word_cnt_per_block * sizeof(typename FilterType::word_t) * 8;
          const auto S = B / c.sector_cnt_per_block;
          const auto fpr = dtl::bloomfilter::fpr(c.m, n, c.k, B, S);
          std::stringstream str;
          str << "\"" << filter_info << "\""
              << "," << c.m
              << "," << (n > 0 ? (c.m*1.0/n) : 0)
              << "," << n
              << "," << sel
              << "," << 0   // DC
              << "," << 0   // false_positives <<<--- DC ?
              << "," << fpr
              << "," << 0.0 // DC
              << "," << 0.0 // DC
              << "," << 1   // single-threaded
              << "," << (force_scalar_code ? "1" : "0")
              << std::endl;
          std::cout << str.str();
        }
      }
      catch (std::exception& err) {
        // Log and go on
        std::stringstream str;
        str << "Aw, snap! Caught: " << err.what() << std::endl;
        std::cout << str.str();
      }
      catch(...) {
        // Log and go on
        std::stringstream str;
        str << "Aw, snap!" << std::endl;
        std::cout << str.str();
      }
    }
    else {
      FilterBenchmark<FilterType>(mode, c.m, c.n, sel, to_add, to_lookup, c.m, c.k, c.word_cnt_per_block, c.sector_cnt_per_block);
    }
  }
  if (c.filter_name == "multiregblocked32" && c.zone_cnt_per_block > 1) {
    using FilterType = dtl::zbbf_32;
    if (fast) { // ------------------------------------------------ <<< FAST RUN
      try {
        FilterType filter = FilterAPI<FilterType>::construct(c.m, c.k, c.word_cnt_per_block, c.zone_cnt_per_block);
        std::string filter_info = FilterAPI<FilterType>::name(&filter);
        boost::replace_all(filter_info, "\"", "\"\""); // escape JSON for CSV output

        for (auto n : c.n) {
          const auto B = c.word_cnt_per_block * sizeof(typename FilterType::word_t) * 8;
          const auto S = sizeof(typename FilterType::word_t) * 8;
          const auto fpr = dtl::bloomfilter::fpr(c.m, n, c.k, B, S, c.zone_cnt_per_block);
          std::stringstream str;
          str << "\"" << filter_info << "\""
              << "," << c.m
              << "," << (n > 0 ? (c.m*1.0/n) : 0)
              << "," << n
              << "," << sel
              << "," << 0   // DC
              << "," << 0   // false_positives <<<--- DC ?
              << "," << fpr
              << "," << 0.0 // DC
              << "," << 0.0 // DC
              << "," << 1   // single-threaded
              << "," << (force_scalar_code ? "1" : "0")
              << std::endl;
          std::cout << str.str();
        }
      }
      catch (std::exception& err) {
        // Log and go on
        std::stringstream str;
        str << "Aw, snap! Caught: " << err.what() << std::endl;
        std::cout << str.str();
      }
      catch(...) {
        // Log and go on
        std::stringstream str;
        str << "Aw, snap!" << std::endl;
        std::cout << str.str();
      }
    }
    else {
      FilterBenchmark<FilterType>(mode, c.m, c.n, sel, to_add, to_lookup, c.m, c.k, c.word_cnt_per_block, c.zone_cnt_per_block);
    }
  }

  if (c.filter_name == "multiregblocked64" && c.zone_cnt_per_block == 1) {
    using FilterType = dtl::bbf_64;
    if (fast) { // ------------------------------------------------ <<< FAST RUN
      try {
        FilterType filter = FilterAPI<FilterType>::construct(c.m, c.k, c.word_cnt_per_block, c.sector_cnt_per_block);
        std::string filter_info = FilterAPI<FilterType>::name(&filter);
        boost::replace_all(filter_info, "\"", "\"\""); // escape JSON for CSV output

        for (auto n : c.n) {
          const auto B = c.word_cnt_per_block * sizeof(typename FilterType::word_t) * 8;
          const auto S = B / c.sector_cnt_per_block;
          const auto fpr = dtl::bloomfilter::fpr(c.m, n, c.k, B, S);
          std::stringstream str;
          str << "\"" << filter_info << "\""
              << "," << c.m
              << "," << (n > 0 ? (c.m*1.0/n) : 0)
              << "," << n
              << "," << sel
              << "," << 0   // DC
              << "," << 0   // false_positives <<<--- DC ?
              << "," << fpr
              << "," << 0.0 // DC
              << "," << 0.0 // DC
              << "," << 1   // single-threaded
              << "," << (force_scalar_code ? "1" : "0")
              << std::endl;
          std::cout << str.str();
        }
      }
      catch (std::exception& err) {
        // Log and go on
        std::stringstream str;
        str << "Aw, snap! Caught: " << err.what() << std::endl;
        std::cout << str.str();
      }
      catch(...) {
        // Log and go on
        std::stringstream str;
        str << "Aw, snap!" << std::endl;
        std::cout << str.str();
      }
    }
    else {
      FilterBenchmark<FilterType>(mode, c.m, c.n, sel, to_add, to_lookup, c.m, c.k, c.word_cnt_per_block, c.sector_cnt_per_block);
    }
  }
  if (c.filter_name == "multiregblocked64" && c.zone_cnt_per_block > 1) {
    using FilterType = dtl::zbbf_64;
    if (fast) { // ------------------------------------------------ <<< FAST RUN
      try {
        FilterType filter = FilterAPI<FilterType>::construct(c.m, c.k, c.word_cnt_per_block, c.zone_cnt_per_block);
        std::string filter_info = FilterAPI<FilterType>::name(&filter);
        boost::replace_all(filter_info, "\"", "\"\""); // escape JSON for CSV output

        for (auto n : c.n) {
          const auto B = c.word_cnt_per_block * sizeof(typename FilterType::word_t) * 8;
          const auto S = sizeof(typename FilterType::word_t) * 8;
          const auto fpr = dtl::bloomfilter::fpr(c.m, n, c.k, B, S, c.zone_cnt_per_block);
          std::stringstream str;
          str << "\"" << filter_info << "\""
              << "," << c.m
              << "," << (n > 0 ? (c.m*1.0/n) : 0)
              << "," << n
              << "," << sel
              << "," << 0   // DC
              << "," << 0   // false_positives <<<--- DC ?
              << "," << fpr
              << "," << 0.0 // DC
              << "," << 0.0 // DC
              << "," << 1   // single-threaded
              << "," << (force_scalar_code ? "1" : "0")
              << std::endl;
          std::cout << str.str();
        }
      }
      catch (std::exception& err) {
        // Log and go on
        std::stringstream str;
        str << "Aw, snap! Caught: " << err.what() << std::endl;
        std::cout << str.str();
      }
      catch(...) {
        // Log and go on
        std::stringstream str;
        str << "Aw, snap!" << std::endl;
        std::cout << str.str();
      }
    }
    else {
      FilterBenchmark<FilterType>(mode, c.m, c.n, sel, to_add, to_lookup, c.m, c.k, c.word_cnt_per_block, c.zone_cnt_per_block);
    }
  }

}


//===----------------------------------------------------------------------===//
// PERFORMANCE measurement
// Run all configurations one after another. (exclusive mode)
//===----------------------------------------------------------------------===//
void execute_performance(const std::vector<config_t>& configs,
                         const double sel,
                         const std::vector<uint32_t, dtl::mem::numa_allocator<uint32_t>>& to_add,
                         const std::vector<uint32_t, dtl::mem::numa_allocator<uint32_t>>& to_lookup) {
  auto time_start = std::chrono::system_clock::now();
  for (std::size_t i = 0; i < configs.size(); i++) {
    const auto c = configs[i];
    execute(benchmark_mode_t::EXCLUSIVE, c, sel, to_add, to_lookup);
    if ((i+1) % 10 == 0) {
      // Estimate time until completion.
      auto now = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed_seconds = now - time_start;
      f64 avg_sec_per_config = elapsed_seconds.count() / i;
      u64 remaining_sec = avg_sec_per_config * (configs.size() - i);
      u64 h = (remaining_sec / 3600);
      u64 m = (remaining_sec % 3600) / 60;
      std::stringstream str;
      str << "Progress of current run: [" << (i + 1) << "/" << configs.size() << "]";
      str << " - est. time until completion: " << h << "h " << m << "m" << std::endl;
      std::cerr << str.str();

      bool paused = false;
      auto paused_begin = std::chrono::system_clock::now();
      {
        std::ifstream pause_file("/tmp/pause");
        if (pause_file.good()) {
          paused = true;

          std::cerr << "Benchmark paused. Delete file /tmp/pause to continue." << std::endl;
        }
      }
      while (true) {
        std::ifstream pause_file("/tmp/pause");
        if (pause_file.good()) {
          std::this_thread::sleep_for(std::chrono::seconds(10));
        }
        else {
          break;
        }
      }
      auto paused_end = std::chrono::system_clock::now();
      if (paused) {
        time_start += paused_end - paused_begin;
      }


    }
  }
  auto time_end = std::chrono::system_clock::now();
  std::chrono::duration<double> duration = time_end - time_start;
  u64 elapsed_sec = duration.count();
  std::cerr << "Performance benchmark completed after " << (elapsed_sec / 3600) << "h " << ((elapsed_sec % 3600) / 60) << "m" << std::endl;
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
// PRECISION measurement
// Run as many configurations as possible in parallel (ignoring NUMA effects).
//===----------------------------------------------------------------------===//
void execute_precision(const std::vector<config_t>& configs,
                       const double sel,
                       const std::vector<uint32_t, dtl::mem::numa_allocator<uint32_t>>& to_add,
                       const std::vector<uint32_t, dtl::mem::numa_allocator<uint32_t>>& to_lookup) {
  const auto time_start = std::chrono::system_clock::now();
  // Use all available cores, unless specified otherwise.
  const auto thread_cnt = dtl::env<$u64>::get("THREAD_CNT_HI", cpu_mask.count());
  const auto config_cnt = configs.size();
  const std::size_t min_inc = 1;//32;
//  const std::size_t max_inc = 256;

  std::atomic<std::size_t> cntr { 0 };
  auto thread_fn = [&](u32 thread_id) {
    while (true) {
      // Grab work.
      const auto inc = min_inc;
      const std::size_t config_idx_begin = cntr.fetch_add(inc);
      const std::size_t config_idx_end = std::min(config_idx_begin + inc, config_cnt);
      if (config_idx_begin >= config_cnt) break;

      for (std::size_t ci = config_idx_begin; ci < config_idx_end; ci++) {
        execute(benchmark_mode_t::SHARED, configs[ci], sel, to_add, to_lookup);
      }

      if (thread_id == 0) {
        u64 i = cntr;
        u64 r = std::min(config_cnt, config_cnt - i);
        // Estimate time until completion.
        const auto now = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = now - time_start;
        f64 avg_sec_per_config = elapsed_seconds.count() / i;
        u64 remaining_sec = avg_sec_per_config * r;
        u64 h = (remaining_sec / 3600);
        u64 m = (remaining_sec % 3600) / 60;
        std::stringstream str;
        str << "Progress of current run: [" << (i + 1) << "/" << config_cnt << "]";
        str << " - est. time until completion: " << h << "h " << m << "m" << std::endl;
        std::cerr << str.str();
      }
    }
  };
  dtl::run_in_parallel(thread_fn, cpu_mask, thread_cnt);
  auto time_end = std::chrono::system_clock::now();
  std::chrono::duration<double> duration = time_end - time_start;
  u64 elapsed_sec = duration.count();
  std::cerr << "Precision benchmark completed after " << (elapsed_sec / 3600) << "h " << ((elapsed_sec % 3600) / 60) << "m" << std::endl;
}
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
int main(int argc, char * argv[]) {

  // Initialize thread_id -> NUMA node_id mapping.
  const auto thread_cnt = dtl::env<$u64>::get("THREAD_CNT_HI", cpu_mask.count());
  std::cerr << "thread_cnt=" << thread_cnt << std::endl;
  thread_id_to_node_id_map.resize(thread_cnt, 0);
  auto thread_fn = [&](const uint32_t thread_id) {
    const auto cpu_node_id = dtl::mem::get_node_of_cpu(
        dtl::this_thread::get_cpu_affinity().find_first());
    const auto mem_node_id = dtl::mem::hbm_available()
        ? dtl::mem::get_nearest_hbm_node(cpu_node_id)
        : cpu_node_id;
    thread_id_to_node_id_map[thread_id] = mem_node_id;
  };
  dtl::run_in_parallel(thread_fn, cpu_mask, thread_cnt);
  std::cerr << "thread id -> node id:" << std::endl;
  for (std::size_t i = 0; i < thread_id_to_node_id_map.size(); i++) {
    std::cerr << " " << i << " -> " << thread_id_to_node_id_map[i] << std::endl;
  }

  const auto DC = 0; // Don't care

  const std::map<std::string, std::string> filter_names {
      {"std",                  "Standard Bloom"}, // costs for positive and negative queries differ
      {"columbia",             "Vectorized Standard Bloom (Columbia University)"}, // costs for positive and negative queries differ
      {"impala",               "Blocked Bloom Impala"},
      {"multiregblocked32",    "Multi register blocked Bloom (32-bit word)"}, // outdated name! "multiregblocked32" supports all kinds of blocked bloom filters.
      {"multiregblocked64",    "Multi register blocked Bloom (64-bit word)"}, // outdated name! "multiregblocked64" supports all kinds of blocked bloom filters.
//      {"regblocked32",         "Register blocked Bloom (32-bit word)"}, // deprecated
//      {"regsecblocked32",      "Register blocked Bloom /w sectorization (32-bit word)"}, // deprecated
//      {"regblocked64",         "Register blocked Bloom (64-bit word)"}, // deprecated
//      {"regsecblocked64",      "Register blocked Bloom /w sectorization (64-bit word)"}, // deprecated
      {"blockedcuckoo",        "Blocked Cuckoo"}, // experimental
      {"cuckoo",               "Cuckoo"},
//      {"dynblocked",           "Generic blocked Bloom filter (no compile-time optimizations)"}, // deprecated
  };


  std::set<std::string> filters_under_test;
  {
    const std::string filters = "," + dtl::env<std::string>::get("FILTERS", "all");
    if (filters == ",all") {
      for (auto f : filter_names) {
        filters_under_test.insert(f.first);
      }
    }
    else {
      for (auto f : filter_names) {
        if (filters.find("," + f.first) != std::string::npos) {
          filters_under_test.insert(f.first);
        }
      }
    }
  }

  if (filters_under_test.size() == 0) {
    std::cerr << "No filters specified. Please set the FILTERS env. variable." << std::endl;
    std::cerr << "Valid options are (colon separated):" << std::endl;
    for (auto fn : filter_names) {
      std::cerr << std::setw(20) << fn.first << fn.second << std::endl;
    }
    std::cerr << std::setw(20) << "all" << std::setw(70) << std::left  << "(default)" << std::endl;

    std::exit(1);
  }

  std::cerr << "Filters under test:" << std::endl;
  for (auto f : filters_under_test) {
    std::cerr << " - " << filter_names.find(f)->second << std::endl;
  }

  //===----------------------------------------------------------------------===//
  // Min/Max filter size in bits
  u64 m_lo = std::max(dtl::env<$u64>::get("M_LO", u64(8 * 1024ull) * 8),
      u64(8 * 1024ull) * 8);           // at least 8 KiB
  u64 m_hi = std::min(dtl::env<$u64>::get("M_HI", u64(256ull) * 1024 * 1024 * 8),
      u64(1024ull) * 1024 * 1024 * 8); // at most 1 GiB
  //===----------------------------------------------------------------------===//

  // all valid filter sizes
  const std::set<$u64> m_s = [&]() {
    std::set<$u64> m_s;
    // power of two filter sizes
    for ($u64 m = dtl::next_power_of_two(m_lo); m <= dtl::next_power_of_two(m_hi); m *= 2) {
      m_s.insert(m);
    }
    // magic filter sizes
    // # powers of two +/- x%
    for ($u64 m = dtl::next_power_of_two(m_lo); m <= dtl::next_power_of_two(m_hi); m *= 2) {
      for (auto percent : {0.01, 0.05, 0.10 }) {
        if ((m + m * percent) <= m_hi) m_s.insert(m + m * percent);
        if ((m - m * percent) >= m_lo) m_s.insert(m - m * percent);
      }
    }
    // # powers of two intermediates
    for ($u64 m = dtl::next_power_of_two(m_lo); m <= dtl::next_power_of_two(m_hi); m *= 2) {
      u64 num_inter_steps = 3;
      u64 step = m / (num_inter_steps + 1);
      for ($u64 i = 1; i <= num_inter_steps; i++) {
        if ((m + i * step) <= m_hi) m_s.insert(m + i * step);
      }
    }
    return m_s;
  }();


  //===----------------------------------------------------------------------===//
  // Bits-per-element rate
  // Note: b = m / n
  const auto b_lo = dtl::env<$u64>::get("BITS_PER_ELEMENT_LO", 4);
  const auto b_hi = dtl::env<$u64>::get("BITS_PER_ELEMENT_HI", 32) > 0
      ? dtl::env<$u64>::get("BITS_PER_ELEMENT_HI", 32)
      : std::numeric_limits<u64>::max();
  //===----------------------------------------------------------------------===//

  //===----------------------------------------------------------------------===//
  // N boundaries
  std::this_thread::sleep_for(std::chrono::seconds {1});
  u64 n_lo = dtl::next_power_of_two(dtl::env<$u64>::get("N_LO", 1ull << 10));
  u64 n_lo_log2 = dtl::log_2(n_lo);
  u64 n_hi = dtl::next_power_of_two(dtl::env<$u64>::get("N_HI", 1ull << 28));
  u64 n_hi_log2 = dtl::log_2(n_hi);
  //===----------------------------------------------------------------------===//

  // all valid n's
  const std::set<$u64> n_s = [&]() {
    std::set<$u64> n_s;

    for ($u64 n_log2 = n_lo_log2; n_log2 <= n_hi_log2; n_log2++) {
//      const std::vector<$f64> exp {
//          n_log2 +  0 * 0.0625,
//          n_log2 +  1 * 0.0625,
//          n_log2 +  2 * 0.0625,
//          n_log2 +  3 * 0.0625,
//          n_log2 +  4 * 0.0625,
//          n_log2 +  5 * 0.0625,
//          n_log2 +  6 * 0.0625,
//          n_log2 +  7 * 0.0625,
//          n_log2 +  8 * 0.0625,
//          n_log2 +  9 * 0.0625,
//          n_log2 + 10 * 0.0625,
//          n_log2 + 11 * 0.0625,
//          n_log2 + 12 * 0.0625,
//          n_log2 + 13 * 0.0625,
//          n_log2 + 14 * 0.0625,
//          n_log2 + 15 * 0.0625,
//          n_log2 + 16 * 0.0625,
//      };
//      const std::vector<$f64> exp {
//          n_log2 + 0 * 0.125,
//          n_log2 + 1 * 0.125,
//          n_log2 + 2 * 0.125,
//          n_log2 + 3 * 0.125,
//          n_log2 + 4 * 0.125,
//          n_log2 + 5 * 0.125,
//          n_log2 + 6 * 0.125,
//          n_log2 + 7 * 0.125,
//          n_log2 + 8 * 0.125,
//      };
      const std::vector<$f64> exp {
          n_log2 + 0 * 0.25,
          n_log2 + 1 * 0.25,
          n_log2 + 2 * 0.25,
          n_log2 + 3 * 0.25,
          n_log2 + 4 * 0.25,
      };

      for (auto e : exp) {
        u64 n = std::pow(2, e);
        if ((n * b_lo) > m_hi) continue; // make sure to not exceed the max filter size
        n_s.insert(n);
      }
    }
    return n_s;
  }();

  std::cerr << "---------------------" << std::endl;
  std::cerr << "N's:" << std::endl;
  for (auto n : n_s) {
    std::cerr << n << " " << std::endl;
  }
  std::cerr << "---------------------" << std::endl;


  // Group by m (allows to construct a filter of size m once and test multiple n's)
  // -- reduce the run time
  std::map<$u64, std::set<$u64>> m_n_map;
  std::map<$u64, std::set<$u64>> n_m_map;
  for (auto n: n_s) {
    for (auto m : m_s) {
      f64 bpe = m / n;
      if (bpe >= b_lo && bpe <= b_hi) {
        if (m_n_map.count(m) == 0) {
          m_n_map.insert(std::make_pair(m, std::set<$u64> { n }));
        }
        else {
          m_n_map[m].insert(n);
        }
        if (n_m_map.count(n) == 0) {
          n_m_map.insert(std::make_pair(n, std::set<$u64> { m }));
        }
        else {
          n_m_map[n].insert(m);
        }
      }
    }
  }


  // NOTE: To accelerate measurements, we separate PERFORMANCE and PRECISION measurements.
  //       The results are supposed to be joined later on.

  // Dump the (m,n) pairs and count the number of performance and precision measurements.
  $u64 perf_run_cntr = 0;
  $u64 prec_run_cntr = 0;

  $u64 m_max = 0;
  for (auto m : m_s) {
    perf_run_cntr++;
    if (m_max < m) m_max = m;
    std::cerr << m << " (" << (m/1024/8) <<" KiB): ";
    for (auto n : m_n_map[m]) {
      prec_run_cntr++;
      std::cerr << n << " (" << (m*1.0)/n << " Bpe) ";
    }
    std::cerr << std::endl;
  }
  std::cerr << "-----------------" << std::endl;

  $u64 n_max = 0;
  for (auto n : n_s) {
    std::cerr << n << ": ";
    if (n_max < n) n_max = n;
    for (auto m : n_m_map[n]) {
      std::cerr << m << " (" << (m*1.0)/n << " Bpe) ";
    }
    std::cerr << std::endl;
  }


  std::cerr << "m_max: " << m_max << " (" << (m_max/1024.0/8) << " KiB)" << std::endl;
  std::cerr << "n_max: " << n_max << std::endl;

  f64 sel = dtl::env<$f64>::get("SEL", 0.0); // 0.0 = only negative queries


  //===----------------------------------------------------------------------===//
  // k
  const auto k_lo = dtl::env<$u64>::get("K_LO", 1);
  const auto k_hi = dtl::env<$u64>::get("K_HI", 16);
  //===----------------------------------------------------------------------===//

   // SIMD calibration required?
  $u1 calibrate_bbf_32 = false;
  $u1 calibrate_bbf_64 = false;
  $u1 calibrate_zbbf_32 = false;
  $u1 calibrate_zbbf_64 = false;
  $u1 calibrate_cf = false;

  //===----------------------------------------------------------------------===//
  auto create_configs = [&](std::vector<config_t>& configurations,
                            const std::size_t m, const std::set<uint64_t> n) { // n values must be in ascending order (guaranteed by std::set)

    //===----------------------------------------------------------------------===//
    // Standard Bloom filter
    //===----------------------------------------------------------------------===//
    if (filters_under_test.count("std")) {
      if ((dtl::is_power_of_two(m) && pow2_addressing) || (!dtl::is_power_of_two(m) && magic_addressing)) {
        for (uint32_t k = k_lo; k <= k_hi; k++) {
          configurations.emplace_back(config_t {"std", m, n, k});
        }
      }
    }


    //===----------------------------------------------------------------------===//
    // Vectorized Bloom filter (Columbia University)
    //===----------------------------------------------------------------------===//
    if (dtl::is_power_of_two(m) && pow2_addressing) { // supports only powers of two
      if (filters_under_test.count("columbia")) {
        for (uint32_t k = k_lo; k <= k_hi; k++) {
          configurations.emplace_back(config_t {"columbia", m, n, k});
        }
      }
    }


    //===----------------------------------------------------------------------===//
    // Cuckoo filter
    //===----------------------------------------------------------------------===//
    const uint32_t cuckoo_tag_size_bits_lo = dtl::env<$u64>::get("CUCKOO_TAG_SIZE_BITS_LO", 4);
    const uint32_t cuckoo_tag_size_bits_hi = dtl::env<$u64>::get("CUCKOO_TAG_SIZE_BITS_HI", 32);

    const uint32_t cuckoo_associativity_lo = dtl::env<$u64>::get("CUCKOO_ASSOCIATIVITY_LO", 1);
    const uint32_t cuckoo_associativity_hi = dtl::env<$u64>::get("CUCKOO_ASSOCIATIVITY_HI", 4);

    if (filters_under_test.count("cuckoo")) {
      calibrate_cf = true;
      if ((dtl::is_power_of_two(m) && pow2_addressing) || (!dtl::is_power_of_two(m) && magic_addressing)) {
        for (uint32_t T = cuckoo_tag_size_bits_lo; T <= cuckoo_tag_size_bits_hi; T += 2) {
          for (uint32_t S = cuckoo_associativity_lo; S <= cuckoo_associativity_hi; S *= 2) {
            configurations.emplace_back(config_t {"cuckoo", m, n, DC, DC, DC, DC, T, S });
          }
        }
      }
    }


    //===----------------------------------------------------------------------===//
    // Blocked Cuckoo filter
    //===----------------------------------------------------------------------===//
    const uint32_t cuckoo_block_size_bytes_lo = dtl::env<$u64>::get("CUCKOO_BLOCK_SIZE_BYTES_LO", 64); // TODO 32
    const uint32_t cuckoo_block_size_bytes_hi = dtl::env<$u64>::get("CUCKOO_BLOCK_SIZE_BYTES_HI", 128);

    if (filters_under_test.count("blockedcuckoo")) {
      if ((dtl::is_power_of_two(m) && pow2_addressing) || (!dtl::is_power_of_two(m) && magic_addressing)) {
        for (uint32_t B = cuckoo_block_size_bytes_lo; B <= cuckoo_block_size_bytes_hi; B *= 2) {
          for (uint32_t T = cuckoo_tag_size_bits_lo; T <= cuckoo_tag_size_bits_hi; T += 2) {
            for (uint32_t S = cuckoo_associativity_lo; S <= cuckoo_associativity_hi; S += 2) {
              configurations.emplace_back(config_t {"blockedcuckoo", m, n, DC, DC, DC, B, T, S});
            }
          }
        }
      }
    }


    //===----------------------------------------------------------------------===//
    // Blocked Bloom filter (Impala)
    //===----------------------------------------------------------------------===//
    if (filters_under_test.count("impala")) {
      if (dtl::is_power_of_two(m) && pow2_addressing) { // supports only k=8 and m of powers of two
        configurations.emplace_back(config_t {"impala", m, n, DC, DC, DC, DC, DC});
      }
    }


    //===----------------------------------------------------------------------===//
    // Blocked Bloom filter (Amsterdam)
    // block size scales from register to cache-line size
    //===----------------------------------------------------------------------===//
    const uint32_t multi_word_cnt_lo = dtl::env<$u64>::get("MULTI_WORD_CNT_LO",  1);
    const uint32_t multi_word_cnt_hi = dtl::env<$u64>::get("MULTI_WORD_CNT_HI", 16);

    const uint32_t multi_sector_cnt_lo = dtl::env<$u64>::get("MULTI_SECTOR_CNT_LO", 1);
    const uint32_t multi_sector_cnt_hi = dtl::env<$u64>::get("MULTI_SECTOR_CNT_HI", 4 * multi_word_cnt_hi);

    // z (number of zones)
    const uint32_t z_lo = dtl::env<$u64>::get("Z_LO", 1);
    const uint32_t z_hi = dtl::env<$u64>::get("Z_HI", 8);
    //===----------------------------------------------------------------------===//

    if (filters_under_test.count("multiregblocked32")) {
      if ((dtl::is_power_of_two(m) && pow2_addressing) || (!dtl::is_power_of_two(m) && magic_addressing)) {
        if (z_lo == 1) {
          const uint32_t z = 1;
          calibrate_bbf_32 = true;
          for (uint32_t W = multi_word_cnt_lo; W <= multi_word_cnt_hi; W *= 2) {
            for (uint32_t S = multi_sector_cnt_lo; S <= multi_sector_cnt_hi; S *= 2) {
//#warning remove me!!!
//            if (S != 1 && S < W) continue;
//              if (S == 1 || S == W) continue;
              for (uint32_t k = k_lo; k <= k_hi; k++) {
                configurations.emplace_back(config_t {"multiregblocked32", m, n, k, W, S, DC, DC, DC, z});
              }
            }
          }
        }

        // add zoned configurations
        if (z_hi > 1) {
          calibrate_zbbf_32 = true;
          for (uint32_t W = std::max(4u, multi_word_cnt_lo); W <= multi_word_cnt_hi; W *= 2) { // at least 4 words are required to enable zones
            const uint32_t S = W; // each word is a sector
            for (uint32_t z = std::max(2u, z_lo); z <= z_hi; z *=2) { // at least 2 zones (note: 1 zone == standard sectorization)
              for (uint32_t k = k_lo; k <= k_hi; k++) {
                if ((k % z) != 0) continue;
                configurations.emplace_back(config_t {"multiregblocked32", m, n, k, W, S, DC, DC, DC, z});
              }
            }
          }
        }

      }
    }

    if (filters_under_test.count("multiregblocked64")) {
      if ((dtl::is_power_of_two(m) && pow2_addressing) || (!dtl::is_power_of_two(m) && magic_addressing)) {
        if (z_lo == 1) {
          const uint32_t z = 1;
          calibrate_bbf_64 = true;
          for (uint32_t W = multi_word_cnt_lo; W <= multi_word_cnt_hi; W *= 2) {
            for (uint32_t S = multi_sector_cnt_lo; S <= multi_sector_cnt_hi; S *= 2) {
//#warning remove me!!!
//            if (S != 1 && S < W) continue;
//            if (S == 1 || S == W) continue;
              for (uint32_t k = k_lo; k <= k_hi; k++) {
                configurations.emplace_back(config_t {"multiregblocked64", m, n, k, W, S, DC, DC, DC, z});
              }
            }
          }
        }

        // add zoned configurations
        if (z_hi > 1) {
          calibrate_zbbf_64 = true;
          for (uint32_t W = std::max(4u, multi_word_cnt_lo); W <= multi_word_cnt_hi; W *= 2) { // at least 4 words are required to enable zones
            const uint32_t S = W; // each word is a sector
            for (uint32_t z = std::max(2u, z_lo); z <= z_hi; z *=2) { // at least 2 zones (note: 1 zone == standard sectorization)
              for (uint32_t k = k_lo; k <= k_hi; k++) {
                if ((k % z) != 0) continue;
                configurations.emplace_back(config_t {"multiregblocked64", m, n, k, W, S, DC, DC, DC, z});
              }
            }
          }
        }

      }
    }
  };
  //===----------------------------------------------------------------------===//

  // Prepare configurations for --precision-- measurements
  std::vector<config_t> prec_configurations;
  if (bench_precision) {
    for (std::size_t m : m_s) {
        create_configs(prec_configurations, m, m_n_map[m]);
    }
  }
  if (prec_configurations.size() > 1) {
    std::cerr << "Prepared " << prec_configurations.size() << " configurations for precision benchmarking." << std::endl;
  }

  // Prepare configurations for --performance-- measurements
  std::vector<config_t> perf_configurations;
  if (bench_performance) {
    if (sel > 0.0) {
      for (std::size_t m : m_s) {
        // This is where run time explodes as we need to run (exclusive) performance measurements for multiple n's.
        create_configs(perf_configurations, m, m_n_map[m]);
      }
    }
    else {
      // Branch-free filters
      for (std::size_t m : m_s) {
        create_configs(perf_configurations, m, /* n = */ {0} );
      }
    }
  }
  if (perf_configurations.size() > 1) {
    std::cerr << "Prepared " << perf_configurations.size() << " configurations for performance benchmarking." << std::endl;
  }


  //===----------------------------------------------------------------------===//

  using key_t = uint32_t;
  // Pin main thread to core 0.
  dtl::thread_affinitize(cpu_mask.find_first());

  std::cerr << "sel=" << sel << std::endl;

  // CSV header
  std::cout << "filter"
            << ",m"
            << ",b"
            << ",n"
            << ",sel"
            << ",insert_time_nanos"
            << ",false_positives"
            << ",fpr"
            << ",lookups_per_sec"
            << ",cycles_per_lookup"
            << ",thread_cnt"
            << ",scalar_code" // deprecated
            << std::endl;

  // The total number of repetitions.
  const auto runs = dtl::env<$u64>::get("RUNS", 1);

  for (std::size_t r = 0; r < runs; r++) {

    std::cerr << "Generating data..." << std::endl;
    std::vector<key_t, dtl::mem::numa_allocator<uint32_t>> to_add;

    // allocate probe data interleaved among all active memory nodes (excluding HBM nodes)
    auto node_mask = cpu_mask;
    node_mask.reset();
    for (auto it = cpu_mask.on_bits_begin(); it != cpu_mask.on_bits_end(); it++) {
      node_mask.set(dtl::mem::get_node_of_cpu(*it));
    }
    std::vector<$u32> numa_nodes;
    for (auto it = node_mask.on_bits_begin(); it != node_mask.on_bits_end(); it++) {
      numa_nodes.push_back(*it);
    }
    std::cerr << "Allocating probe data on NUMA node(s):";
    for (auto nid : numa_nodes) std::cerr << " " << nid;
    std::cerr << std::endl;

    dtl::mem::allocator_config alloc_config = dtl::mem::allocator_config::on_node(numa_nodes);
    dtl::mem::numa_allocator<uint32_t> alloc(alloc_config);
    std::vector<key_t, dtl::mem::numa_allocator<uint32_t>> to_lookup(alloc);
    gen_data(to_add, to_lookup, (bench_precision | (sel > 0.0)) ? n_max : 0);

    if (force_scalar_code) {
      // setting the unroll factor to 0 enforces the scalar code path
      dtl::bbf_32::force_unroll_factor(0);
      dtl::bbf_64::force_unroll_factor(0);
      dtl::zbbf_32::force_unroll_factor(0);
      dtl::zbbf_64::force_unroll_factor(0);
      dtl::cf::force_unroll_factor(0);
    }
    else {
      //===----------------------------------------------------------------------===//
      // Set or calibrate the SIMD unrolling factor (applies for BBF and CF).
      const auto unroll_factor = dtl::env<$i64>::get("SIMD_UNROLL_FACTOR", -1);
      if (dtl::env<$u64>::get("SIMD_CALIBRATION", 1) == 1
          && unroll_factor != -1) {
        std::cerr << "Conflicting benchmark settings. Please use either "
            << "'SIMD_CALIBRATION=1' or 'SIMD_UNROLL_FACTOR'." << std::endl;
        std::exit(1);
      }
      if (dtl::env<$u64>::get("SIMD_CALIBRATION", 1)) {
        if (calibrate_bbf_32) dtl::bbf_32::calibrate();
        if (calibrate_bbf_64) dtl::bbf_64::calibrate();
        if (calibrate_zbbf_32) dtl::zbbf_32::calibrate();
        if (calibrate_zbbf_64) dtl::zbbf_64::calibrate();
        if (calibrate_cf) dtl::cf::calibrate();
      }
      else if (unroll_factor != -1) {
        std::cerr << "Setting the SIMD unrolling factor to " << unroll_factor
            << std::endl;
        dtl::bbf_32::force_unroll_factor(unroll_factor);
        dtl::bbf_64::force_unroll_factor(unroll_factor);
        dtl::zbbf_32::force_unroll_factor(unroll_factor);
        dtl::zbbf_64::force_unroll_factor(unroll_factor);
        dtl::cf::force_unroll_factor(unroll_factor);
      }
      else {
        std::cerr << "SIMD calibration disabled!" << std::endl;
      }
      //===----------------------------------------------------------------------===//
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    // Shuffle the configurations to better predict the overall runtime of the benchmark.
    if (prec_configurations.size() > 0) {
      std::cerr << "Running precision measurements..." << std::endl;
      std::shuffle(prec_configurations.begin(), prec_configurations.end(), gen);
      execute_precision(prec_configurations, sel, to_add, to_lookup);
    }
    if (perf_configurations.size() > 0) {
      std::cerr << "Running performance measurements..." << std::endl;
      std::shuffle(perf_configurations.begin(), perf_configurations.end(), gen);
      execute_performance(perf_configurations, sel, to_add, to_lookup);
    }

  }

}
