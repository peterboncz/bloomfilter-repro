cmake_minimum_required(VERSION 3.5)
project(cuckoofilter)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -DHAVE_NUMA -fext-numeric-literals -Wall -Wno-ignored-attributes -Wno-deprecated -Wno-endif-labels -Wno-unused-function")

set(CMAKE_VERBOSE_MAKEFILE OFF)

# The Database Template Library (DTL) which includes the SIMDized filter
# implementations presented in the paper. (see dtl/filter/)
add_subdirectory(module/dtl dtl)
include_directories(module/dtl/src)
include_directories(module/dtl/thirdparty)

set(LIBS
        ssl
        crypto
        pthread
        numa
        dl
        )

### build optimized executables for the following architectures
set(ARCHS
        corei7
        core-avx2
        knl
        skx
        )

# replacement for 'add_executable'
function(add_multiarch_executable executable_name files)
    foreach(arch ${ARCHS})
        add_executable(${executable_name}_${arch} ${files})
        # workaround for older compilers which don't know the 'skx' architecture.
        if (arch STREQUAL "skx")
#            set_target_properties(${executable_name}_${arch} PROPERTIES COMPILE_FLAGS "-march=knl -mavx512bw")
            set_target_properties(${executable_name}_${arch} PROPERTIES COMPILE_FLAGS "-march=skylake-avx512")
        else()
            set_target_properties(${executable_name}_${arch} PROPERTIES COMPILE_FLAGS "-march=${arch}")
        endif()
        set(CMAKE_EXE_LINKER_FLAGS " -static")
        target_link_libraries(${executable_name}_${arch} dtl_${arch} ${LIBS} -static-libstdc++)
    endforeach(arch ${ARCHS})
endfunction(add_multiarch_executable)


include_directories(src)

set(BENCHMARK_SOURCE_FILES # headers are included to enable syntax highlighting in CLion

        # unified API for all filters
        benchmarks/filter/api.hpp
        benchmarks/filter/blocked_bloom_multiword.hpp
        benchmarks/filter/blocked_bloom_impala.hpp
        benchmarks/filter/blocked_cuckoo.hpp
        benchmarks/filter/bloom.hpp
        benchmarks/filter/bloom_columbia.hpp
        benchmarks/filter/cuckoofilter.hpp
        benchmarks/filter/cuckoo_fan.hpp
        benchmarks/filters.hpp

        # benchmark helpers
        benchmarks/util.hpp
        benchmarks/random.h
        benchmarks/timing.h

        # cuckoofilter (Fan et al.)
        src/bitsutil.h
        src/cuckoofilter.h
        src/debug.h
        src/hashutil.cc
        src/hashutil.h
        src/packedtable.h
        src/permencoding.h
        src/printutil.h
        src/simd-block.h
        src/singletable.h
        src/bettertable.h

        # vectorized bloom filter (columbia)
        src/vbf.hpp
        src/vbf.cpp

        # main
        benchmarks/skyline.cpp
        )

# the benchmark runner
add_multiarch_executable(benchmark "${BENCHMARK_SOURCE_FILES}")

# helper
add_executable(n_filter "benchmarks/n_filter.cpp")
target_link_libraries(n_filter ${LIBS} -static-libstdc++)

add_executable(get_cache_size "benchmarks/get_cache_size.cpp")
target_link_libraries(get_cache_size ${LIBS} -static-libstdc++)

# copy shell scripts
configure_file(scripts/benchmark.sh benchmark.sh COPYONLY)
configure_file(scripts/determine_arch.sh determine_arch.sh COPYONLY)
configure_file(scripts/aggr_results.sh aggr_results.sh COPYONLY)
configure_file(scripts/summary.sh summary.sh COPYONLY)

# copy tex files
set(TEX_FILES
        tex/plot_skyline_bbf_config.tex
        tex/plot_skyline_cf_config.tex
        tex/plot_skyline_overview.tex
        tex/summary.tex
        )
file(COPY ${TEX_FILES} DESTINATION tex)
