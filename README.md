Bloom and Cuckoo Filter Benchmark
=================================

This repo contains the benchmark runner that was used to evaluate
 Bloom and Cuckoo filters for the VLDB'19 paper  [*Performance-Optimal Filtering:
 Bloom Overtakes Cuckoo at High Throughput*](http://www.vldb.org/pvldb/vol12/p502-lang.pdf).

The repo is based on the Cuckoo filter repo and includes a slightly modified
 version of the Cuckoo filter as presented in the ACM CoNEXT'14 paper
 [*Cuckoo Filter: Practically Better Than Bloom*](http://www.cs.cmu.edu/~binfan/papers/conext14_cuckoofilter.pdf).
If you are looking for the latest version of the Cuckoo filter, please refer to
 [https://github.com/efficient/cuckoofilter](https://github.com/efficient/cuckoofilter).

Further we include a copy of the Bloom filter implementation from the
 [Impala](https://impala.apache.org/) database system (see 'src/simd-block.h')
 and the [vectorized Bloom filter](http://www.cs.columbia.edu/~orestis/vbf.c)
 as presented in the DaMoN'14 paper
 [*Vectorized Bloom Filters for Advanced SIMD Processors*](http://www.cs.columbia.edu/~orestis/damon14.pdf).

Our SIMD-optimized implementations of Bloom and Cuckoo filters are included
 as a git submodule. The source code can be found in the GitHub repo
 [bloomfilter-bsd](https://github.com/peterboncz/bloomfilter-bsd).


### Erratum
Post-publication an error was found (and fixed) in the collision resolution of
 cuckoo filters with arbitrarily sized tables.
We refer to our blog post
["Cuckoo Filters with arbitrarily sized tables"](https://databasearchitects.blogspot.com/2019/07/cuckoo-filters-with-arbitrarily-sized.html) for details.


Using the Code
--------------
### Prerequisites
* A C++14 compliant compiler; only GCC has been tested.
* [CMake](http://www.cmake.org/) version 3.5 or later.
* The [Boost C++ Libraries](https://www.boost.org/), version 1.58 or later.
* A Linux environment (including the BASH shell and the typical GNU tools).
* SQLite version 3.x
* a TeX distribution, e.g. TeX Live (optional)


### Repository structure
* `benchmarks/`: the benchmark runner
* `module/dtl/`: git submodule for the SIMD-optimized filter implementations
 * In particular, our Bloom filter implementation can be found in `./filter/blocked_bloomfilter/` and our Cuckoo implementation in
 `./filter/cuckoofilter/`
* `scripts/`: several shell scripts that drive the benchmark
* `src/`: the C++ header and implementation of the (original) cuckoo filter, the
   Impala and vectorized Bloom filters
* `tex/`: LaTeX files to typeset the results

### Building
```
git clone git@github.com:peterboncz/bloomfilter-repro.git
cd bloomfilter-repro
git submodule update --remote --recursive --init
mkdir build
cd build/
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8 n_filter
make -j 8 get_cache_size
make -j 8 benchmark_`./determine_arch.sh`
```
The benchmark runner can be compiled for the following architectures:

| Architecture | Description                                                                              |
| ------------ | ---------------------------------------------------------------------------------------- |
| `corei7`     | targets pre-AVX2 processor generations. All SIMD optimizations are disabled.             |
| `core-avx2`  | targets Intel Haswell (or later) and AMD Ryzen processors with the AVX2 instruction set. |
| `knl`        | targets Intel Knights Landing (KNL) processor with the AVX-512F instruction set.         |
| `skx`        | targets Intel Skylake-X (or later) processors with the AVX-512F/BW instruction set.      |

### Benchmarking

For a quick start, we provide a *scripted* benchmark which automatically
 performs several performance measurements and imports the results into a
 SQLite database. Optionally a summary sheet is generated.

The following scripts need to be executed in the given order:
```
./benchmark.sh
./aggr_results.sh
./summary.sh
```
The `benchmark.sh` script performs the actual measurements and stores the CSV results in
 the directory `./results`.
The `aggr_results.sh` script imports the raw results into a SQLite database
 stored in `./results/skyline.sqlite3`.
Optionally, the `summary.sh` script typesets a summary PDF.

To perform other analyses, we refer to the source code of the scripts
 mentioned above.
Further details on the output format and
 the benchmark options can be found [here](BENCHMARK.md).

Related Work
------------

* [Morton Filter](https://github.com/AMDComputeLibraries/morton_filter)
> A Morton filter is a modified cuckoo filter [...] that is optimized for bandwidth-constrained systems.

* [Fluid Co-Processing](https://github.com/t1mm3/fluid_coprocessing)




Licenses
--------

* The [Cuckoo filter](https://github.com/efficient/cuckoofilter) and the
   [Impala](https://impala.apache.org/) Bloom filter implementation are licensed
   under the Apache License, Version 2.0.
* [Vectorized Bloom filters](http://www.cs.columbia.edu/~orestis/vbf.c) are
   licensed under the 2-clause BSD license.
* Our [SIMD-optimized implementations](https://github.com/peterboncz/bloomfilter-bsd)
   are dual licensed under the Apache License, Version 2.0 and the 3-clause BSD
   license.  
