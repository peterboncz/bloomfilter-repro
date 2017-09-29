Benchmark
---------
The benchmark runner is able to evaluate several types of filters thereby
 varying
 * the individual filter parameters,
 * the filter sizes,
 * the problem sizes (number of elements to insert), and the
 * parallelization level (multi-threading).

By design, the runner explores the entire space spanned by the different
 settings, unless the user manually narrows one or more parameters.
For that the runner offers several options that are passed as environment
 variables (a detailed list can be found below).

The following example shows how the parameter *k* is limited to 4 <= k <=8.
```
K_LO=4 K_HI=8 ./benchmark_skx 2> results.errout | tee results.out
```
The result is written as CSV to stdout and additional information is written
 to stderr, such as the estimated time until completion.


## Output Format

The CSV output consists of 12 columns:

| Column              | Type      | Description                          |
| ------------------- | --------- | ------------------------------------ |
| `filter`            | `JSON`    | filter type (see below)              |
| `m`                 | `integer` | filter size in bits                  |
| `b`                 | `float`   | bits per key                         |
| `n`                 | `integer` | number of inserted keys              |
| `sel`               | `float`   | probability of a true hit            |
| `insert_time_nanos` | `-`       | _deprecated_                         |
| `false_positives`   | `integer` | number of false positives            |
| `FPR`               | `float`   | false positive rate                  |
| `lookups_per_sec`   | `float`   | number of lookups per seconds        |
| `cycles_per_lookup` | `float`   | avg. number of CPU cycles per lookup |
| `thread_cnt`        | `integer` | number of threads                    |
| `scalar_code`       | `-`       | _deprecated_                         |

The `filter` attribute contains all relevant information regarding the filter
 under test.
The following two examples explain the structure of the JSON object:

```javascript
{
   "name":"blocked_bloom_multiword", // filter implementation
   "size":4096, // filter size in bytes
   "word_size":4, // word size in bytes
   "k":8, // number of hash functions
   "w":8, // number of words per block
   "s":8, // number of sectors per block
   "z":4, // number of sector groups (zones)
   "u":0, // SIMD unrolling factor
   "e":0, // deprecated
   "addr":"pow2" // addressing mode
}
```

```javascript
{
   "name":"cuckoo", // filter implementation
   "size":4112, // filter size in bytes
   "tag_bits":8, // signature size in bits (aka tag size)
   "associativity":1, // number of signatures per bucket
   "delete_support":false, // deprecated
   "u":4, // SIMD unrolling factor
   "addr":"magic" // addressing mode
}
```
The JSON keys `name` and  `size` apply for all implemenations, `addr` and `u`
 apply only for our SIMD optimized implementations.

## Options

The `FILTERS` option allows for specifying the filter types under test as a
 comma separated list:

| Filters             | Description                                   |
| ------------------- | --------------------------------------------- |
| `all`               | benchmark all filters (default)               |
| `std`               | Standard Bloom filter                         |
| `columbia`          | Vectorized Bloom filter                       |
| `cuckoo`            | Cuckoo filter                                 |
| `impala`            | Impala blocked Bloom filter                   |
| `multiregblocked32` | SIMD blocked Bloom filter (32-bit SIMD lanes) |
| `multiregblocked64` | SIMD blocked Bloom filter (64-bit SIMD lanes) |


The following options give control over the filter and problem sizes and
thus apply for (almost) all filter types:

| Option                | Description                                                     |
| --------------------- | --------------------------------------------------------------- |
| `M_LO`                | the minimum filter size in bits (default: 65536 = 8 KiB)        |
| `M_HI`                | the maximum filter size in bits (default: 2147483648 = 256 MiB) |
| `POW2_ADDR`           | use power of two filter sizes: 0 = no, 1 = yes (default)        |
| `MAGIC_ADDR`          | use arbitrary filter sizes: 0 = no, 1 = yes (default)           |
| `BITS_PER_ELEMENT_LO` | the minimum number of bits per element (default: 4)             |
| `BITS_PER_ELEMENT_HI` | the maximum number of bits per element (default: 32)            |
| `N_LO`                | the minimum number of elements to insert (default: 2^10)        |
| `N_HI`                | the maximum number of elements to insert  (default: 2^28)       |

Note that `columbia` and `impala` only support filter sizes that are powers of
 two. Thus the option `MAGIC_ADDR` does not apply for these filter types.

[//]: # (TODO describe how M and N are chosen)


### Bloom filter

| Option                | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `K_LO`                | the minimum number of hash functions to use (default: 1)   |
| `K_HI`                | the maximum number of hash functions to use  (default: 16) |
| `MULTI_WORD_CNT_LO`   | the minimum number of words per block (default: 1)         |
| `MULTI_WORD_CNT_HI`   | the maximum number of words per block (default: 8)         |
| `MULTI_SECTOR_CNT_LO` | the minimum number of sectors per block (default: 1)       |
| `MULTI_SECTOR_CNT_HI` | the maximum number of sectors per block (default: 32)      |
| `Z_LO`                | the minimum number of zones per block (default: 1)         |
| `Z_HI`                | the maximum number of zones per block (default: 8)         |

### Cuckoo filter

| Option                    | Description                                          |
| ------------------------- | ---------------------------------------------------- |
| `CUCKOO_TAG_SIZE_BITS_LO` | the minimum signature size [bits] (default:  4)      |
| `CUCKOO_TAG_SIZE_BITS_HI` | the maximum signature size [bits] (default: 32)      |
| `CUCKOO_ASSOCIATIVITY_LO` | the minimum number of slots per bucket (default:  1) |
| `CUCKOO_ASSOCIATIVITY_HI` | the maximum number of slots per bucket (default:  4) |

### Multi-Threading

| Option               | Description                                                                         |
| -------------------- | ----------------------------------------------------------------------------------- |
| `THREAD_CNT_LO`      | the minimum number of threads to use (default: 1)                                   |
| `THREAD_CNT_HI`      | the maximum number of threads to use (default: hardware concurrency)                |
| `THREAD_STEP`        | the number of additional threads per run (default: 1)                               |
| `THREAD_STEP_MODE`   | defines how the number of threads increases. 1 = linear, 2 = exponential (default)  |

### Other Options

| Option               | Description                                                                         |
| -------------------- | ----------------------------------------------------------------------------------- |
| `BENCH_PRECISION`    | 0 = disabled, 1 = enabled                                                           |
| `BENCH_PERFORMANCE`  | 0 = disabled, 1 = enabled                                                           |
| `VALIDATION`         | 0 = disabled, 1 = enabled                                                           |
| `FAST`               | uses the FPR formulas rather than measurements                                      |
| `FILTERS`            | specifies the filter under test (see list above)                                    |
| `SEL`                | the filter selectivity (default: 0.0 = only negative queries)                       |
| `RUNS`               | the number of repetitions                                                           |
| `SIMD_UNROLL_FACTOR` | manually set the SIMD unroll factor, 0 = scalar, 1 = SIMD, x>1 = SIMD unrolled by x |
| `SIMD_CALIBRATION`   | automatically determine the unroll factor; 0 = disabled, 1 = enabled (default)      |
