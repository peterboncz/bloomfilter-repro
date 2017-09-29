#!/bin/bash

# Global settings
RESTRICT_TO_SINGLE_SOCKET="yes"
HYPER_THREADING="no"

#-------------------------------------------------------------------------------
# Inspect the environment.
ARCH="unknown"
if [ ! -z "`cat /proc/cpuinfo | grep avx512bw`" ]; then
  ARCH="skx"
elif [ ! -z "`cat /proc/cpuinfo | grep "Xeon Phi"`" ]; then
  ARCH="knl"
elif [ ! -z "`cat /proc/cpuinfo | grep avx2`" ]; then
  ARCH="core-avx2"
else
  ARCH="corei7"
fi
echo "arch=$ARCH"

NUMACTL=`which numactl`
NUMA_CMD=""
THREAD_CNT=0
if [ "$ARCH" != "knl" ] && [ ! -z "$NUMACTL" ]; then
  if [ "$RESTRICT_TO_SINGLE_SOCKET" == "yes" ]; then
    THREAD_CNT=`numactl -H | grep "node 0 cpus:" | sed 's/node 0 cpus: //' | sed 's/ /\n/g' | wc -l`
    NUMA_CMD="numactl -N 0"
  fi
else
  THREAD_CNT=`cat /proc/cpuinfo | grep processor | wc -l`
fi

if [ "$HYPER_THREADING" == "no" ]; then
  THREAD_CNT=$((THREAD_CNT/2))
fi

if [ "$THREAD_CNT" == "0" ]; then
  echo "Error: Could not determine the number of threads to use."
  exit 1
fi
echo "thread_cnt=$THREAD_CNT"

RUNNER="./benchmark_${ARCH}"
if [ ! -f $RUNNER ]; then
  echo "Error: Could not find the executable '$RUNNER'.";
  exit 1
fi
RUNNER="$NUMA_CMD $RUNNER"
echo "cmd=${RUNNER}"

RESULTS_DIR="results"
mkdir $RESULTS_DIR
#-------------------------------------------------------------------------------
# Common benchmark options.
export MULTI_WORD_CNT_LO=1
export MULTI_WORD_CNT_HI=16
export MULTI_SECTOR_CNT_LO=1
export MULTI_SECTOR_CNT_HI=16
export K_LO=1
export K_HI=16

export CUCKOO_ASSOCIATIVITY_LO=1
export CUCKOO_ASSOCIATIVITY_HI=4

export POW2_ADDR=1
export MAGIC_ADDR=1
export FAST=1

export THREAD_CNT_LO=$THREAD_CNT
export THREAD_CNT_HI=$THREAD_CNT
export THREAD_STEP_MODE=1
export THREAD_STEP=1

FILTER_NAMES="cuckoo impala multiregblocked32 multiregblocked64"

# Defaults - may be overwritten later on.
export BENCH_PRECISION=0
export BENCH_PERFORMANCE=0
export SIMD_CALIBRATION=1
export SIMD_UNROLL_FACTOR=-1

function do_benchmark {
  local benchmark_type="unknown"
  local run_ids="1 2 3 4 5"
  if [ "$1" == "precision" ]; then
    benchmark_type="precision"
    run_ids="1"
    BENCH_PRECISION=1
    BENCH_PERFORMANCE=0
    SIMD_CALIBRATION=0
    FAST=1
  else
    benchmark_type="performance"
    BENCH_PRECISION=0
    BENCH_PERFORMANCE=1
    SIMD_CALIBRATION=1
    FAST=0
  fi

  for FILTER_NAME in $FILTER_NAMES; do
    for RUN_ID in $run_ids; do
      local file_basename="results_${benchmark_type}_${FILTER_NAME}_run${RUN_ID}"
      local file_errout="${RESULTS_DIR}/${file_basename}.errout"
      local file_out="${RESULTS_DIR}/${file_basename}.out"
      # Test if the file already exists and whether it is complete.
      if [ -f $file_errout ]; then
        if [ -z "`tail -1 $file_errout | grep "benchmark completed"`" ]; then
          # File exists, but it seems that the benchmark has been interrupted.
          rm $file_errout
          rm $file_out
        fi
      fi
      if [ -f $file_errout ]; then
        echo "Found result file. Skipping this run."
      else
        echo "Running ${benchmark_type} benchmark using filter '${FILTER_NAME}' (run ${RUN_ID})"
        echo " Output is written to: ${file_out}"
        echo " Log is written to:    ${file_errout}"
        RUNS=1 FILTERS=${FILTER_NAME} $RUNNER 2>> $file_errout >> $file_out
      fi
    done # runs
  done # filters
}

#-------------------------------------------------------------------------------
do_benchmark "precision"
do_benchmark "performance"
