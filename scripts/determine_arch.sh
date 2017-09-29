#!/bin/bash
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
echo "$ARCH"
