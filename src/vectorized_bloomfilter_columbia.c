/* Copyright (c) 2014, Orestis Polychroniou
 * Department of Computer Science, Columbia University
 * All rights reserved.
 *
 * Material for research paper:
 *   Venue:  Data Management on New Hardware (DaMoN) 2014
 *   Title:  Vectorized Bloom Filters for Advanced SIMD Processors
 *   Authors:  Orestis Polychroniou (orestis@cs.columbia.edu)
 *             Kenneth A. Ross (kar@cs.columbia.edu)
 *   Affiliation:  Department of Computer Science, Columbia University
 *
 * License:
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *   1. Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *  To compile:
 *     gcc -O3 -o vbf vbf.c -lpthread -lrt -lm -march=core-avx2
 *     icc -O3 -o vbf vbf.c -lpthread -lrt -lm -march=core-avx2
 *
 *  Tested compilers:  GCC 4.7, GCC 4.8, GCC 4.9, ICC 14, ICC 15
 *
 *  Compatible platforms:  GNU/Linux
 *
 *  Command line arguments:
 *    1) outer_tuples
 *          Cardinality of outer probed table (keys & payloads)
 *    2) inner_tuples
 *          Cardinality of inner built table (unique keys only)
 *    3) filter_size
 *          Bloom filter size in bytes and must be power of 2
 *          (e.g. 16K 128K 2M, 4M, ...)
 *    4) hash_functions
 *          Number of hash functions in the filter (up to 6 functions)
 *    5) filter_rate
 *          Percentage of input tuples that correctly qualify
 *    6) threads (optional)
 *          Number of threads with default value all hardware threads
 */

#define _GNU_SOURCE
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
#include <immintrin.h>
#include <x86intrin.h>

#undef _GNU_SOURCE


// thread timing (in nano seconds)
uint64_t thread_time(void)
{
	struct timespec t;
	//assert(clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t) == 0);
	clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
	return t.tv_sec * 1000 * 1000 * 1000 + t.tv_nsec;
}

// number of hardware threads (GNU/Linux specific)
int hardware_threads(void)
{
	char name[64];
	struct stat st;
	int threads = -1;
	do {
		sprintf(name, "/sys/devices/system/cpu/cpu%d", ++threads);
	} while (stat(name, &st) == 0);
	return threads;
}

// bind thread to specific thread (GNU specific)
void bind_thread(int thread_id)
{
	int threads = hardware_threads();
	assert(thread_id >= 0 && thread_id < threads);
	size_t size = CPU_ALLOC_SIZE(threads);
	cpu_set_t *cpu_set = CPU_ALLOC(threads);
	assert(cpu_set != NULL);
	CPU_ZERO_S(size, cpu_set);
	CPU_SET_S(thread_id, size, cpu_set);
	assert(pthread_setaffinity_np(pthread_self(),
	       size, cpu_set) == 0);
	CPU_FREE(cpu_set);
}

// 32-bit uniform random generator called ``Mersenne Twister''
// MUCH more random than rand(), rand_r(), and *rand48*() functions
typedef struct {
	uint32_t num[625];
	size_t index;
} mt32_t;

// initialize random generator
mt32_t *mt32_init(uint32_t seed)
{
	mt32_t *state = malloc(sizeof(mt32_t));
	uint32_t *n = state->num;
	size_t i;
	n[0] = seed;
	for (i = 0 ; i != 623 ; ++i)
		n[i + 1] = 0x6c078965 * (n[i] ^ (n[i] >> 30));
	state->index = 624;
	return state;
}

// next item of random generator
uint32_t mt32_next(mt32_t *state)
{
	uint32_t y, *n = state->num;
	if (state->index == 624) {
		size_t i = 0;
		do {
			y = n[i] & 0x80000000;
			y += n[i + 1] & 0x7fffffff;
			n[i] = n[i + 397] ^ (y >> 1);
			n[i] ^= 0x9908b0df & -(y & 1);
		} while (++i != 227);
		n[624] = n[0];
		do {
			y = n[i] & 0x80000000;
			y += n[i + 1] & 0x7fffffff;
			n[i] = n[i - 227] ^ (y >> 1);
			n[i] ^= 0x9908b0df & -(y & 1);
		} while (++i != 624);
		state->index = 0;
	}
	y = n[state->index++];
	y ^= (y >> 11);
	y ^= (y << 7) & 0x9d2c5680;
	y ^= (y << 15) & 0xefc60000;
	y ^= (y >> 18);
	return y;
}

// scalar Bloom filter probe (loop over functions)
size_t bloom_scalar_soft(int32_t *keys, int32_t *vals, size_t tuples, int32_t *filter,
                         int32_t *factors, size_t functions, uint8_t log_filter_size,
                         int32_t *keys_out, int32_t *vals_out)
{
	size_t i = 0, o = 0;
	uint8_t shift = 32 - log_filter_size;
	while (i != tuples) {
		int32_t key = keys[i];
		size_t f = 0;
		do {
			uint32_t h = key * factors[f++];
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed]"
			:: "r"(filter), "r"(h)
			: "cc" : failed);
		} while (f != functions);
		vals_out[o] = vals[i];
		keys_out[o++] = key;
failed:
		i++;
	}
	return o;
}

// scalar Bloom filter probe (hard-coded by number of functions)
size_t bloom_scalar_hard(int32_t *keys, int32_t *vals, size_t tuples, int32_t *filter,
                         int32_t factors[6], size_t functions, uint8_t log_filter_size,
                         int32_t *keys_out, int32_t *vals_out)
{
	size_t i = 0, o = 0;
	uint8_t shift = 32 - log_filter_size;
	int32_t f1 = factors[0], f2 = factors[1];
	int32_t f3 = factors[2], f4 = factors[3];
	int32_t f5 = factors[4], f6 = factors[5];
	if (functions == 1)
		while (i != tuples) {
			int32_t key = keys[i];
			uint32_t h = key * f1;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_1]"
			:: "r"(filter), "r"(h) : "cc" : failed_1);
			vals_out[o] = vals[i];
			keys_out[o++] = key;
failed_1:
			i++;
		}
	else if (functions == 2)
		while (i != tuples) {
			int32_t key = keys[i];
			uint32_t h = key * f1;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_2]"
			:: "r"(filter), "r"(h) : "cc" : failed_2);
			h = key * f2;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_2]"
			:: "r"(filter), "r"(h) : "cc" : failed_2);
			vals_out[o] = vals[i];
			keys_out[o++] = key;
failed_2:
			i++;
		}
	else if (functions == 3)
		while (i != tuples) {
			int32_t key = keys[i];
			uint32_t h = key * f1;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_3]"
			:: "r"(filter), "r"(h) : "cc" : failed_3);
			h = key * f2;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_3]"
			:: "r"(filter), "r"(h) : "cc" : failed_3);
			h = key * f3;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_3]"
			:: "r"(filter), "r"(h) : "cc" : failed_3);
			vals_out[o] = vals[i];
			keys_out[o++] = key;
failed_3:
			i++;
		}
	else if (functions == 4)
		while (i != tuples) {
			int32_t key = keys[i];
			uint32_t h = key * f1;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_4]"
			:: "r"(filter), "r"(h) : "cc" : failed_4);
			h = key * f2;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_4]"
			:: "r"(filter), "r"(h) : "cc" : failed_4);
			h = key * f3;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_4]"
			:: "r"(filter), "r"(h) : "cc" : failed_4);
			h = key * f4;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_4]"
			:: "r"(filter), "r"(h) : "cc" : failed_4);
			vals_out[o] = vals[i];
			keys_out[o++] = key;
failed_4:
			i++;
		}
	else if (functions == 5)
		while (i != tuples) {
			int32_t key = keys[i];
			uint32_t h = key * f1;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_5]"
			:: "r"(filter), "r"(h) : "cc" : failed_5);
			h = key * f2;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_5]"
			:: "r"(filter), "r"(h) : "cc" : failed_5);
			h = key * f3;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_5]"
			:: "r"(filter), "r"(h) : "cc" : failed_5);
			h = key * f4;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_5]"
			:: "r"(filter), "r"(h) : "cc" : failed_5);
			h = key * f5;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_5]"
			:: "r"(filter), "r"(h) : "cc" : failed_5);
			vals_out[o] = vals[i];
			keys_out[o++] = key;
failed_5:
			i++;
		}
	else if (functions == 6)
		while (i != tuples) {
			int32_t key = keys[i];
			uint32_t h = key * f1;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_6]"
			:: "r"(filter), "r"(h) : "cc" : failed_6);
			h = key * f2;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_6]"
			:: "r"(filter), "r"(h) : "cc" : failed_6);
			h = key * f3;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_6]"
			:: "r"(filter), "r"(h) : "cc" : failed_6);
			h = key * f4;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_6]"
			:: "r"(filter), "r"(h) : "cc" : failed_6);
			h = key * f5;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_6]"
			:: "r"(filter), "r"(h) : "cc" : failed_6);
			h = key * f6;
			h >>= shift;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed_6]"
			:: "r"(filter), "r"(h) : "cc" : failed_6);
			vals_out[o] = vals[i];
			keys_out[o++] = key;
failed_6:
			i++;
		}
	else assert(functions > 0 && functions < 7);
	return o;
}

// constant table with permutation masks
const uint64_t perm[256] = {0x0706050403020100ull,
     0x0007060504030201ull, 0x0107060504030200ull, 0x0001070605040302ull,
     0x0207060504030100ull, 0x0002070605040301ull, 0x0102070605040300ull,
     0x0001020706050403ull, 0x0307060504020100ull, 0x0003070605040201ull,
     0x0103070605040200ull, 0x0001030706050402ull, 0x0203070605040100ull,
     0x0002030706050401ull, 0x0102030706050400ull, 0x0001020307060504ull,
     0x0407060503020100ull, 0x0004070605030201ull, 0x0104070605030200ull,
     0x0001040706050302ull, 0x0204070605030100ull, 0x0002040706050301ull,
     0x0102040706050300ull, 0x0001020407060503ull, 0x0304070605020100ull,
     0x0003040706050201ull, 0x0103040706050200ull, 0x0001030407060502ull,
     0x0203040706050100ull, 0x0002030407060501ull, 0x0102030407060500ull,
     0x0001020304070605ull, 0x0507060403020100ull, 0x0005070604030201ull,
     0x0105070604030200ull, 0x0001050706040302ull, 0x0205070604030100ull,
     0x0002050706040301ull, 0x0102050706040300ull, 0x0001020507060403ull,
     0x0305070604020100ull, 0x0003050706040201ull, 0x0103050706040200ull,
     0x0001030507060402ull, 0x0203050706040100ull, 0x0002030507060401ull,
     0x0102030507060400ull, 0x0001020305070604ull, 0x0405070603020100ull,
     0x0004050706030201ull, 0x0104050706030200ull, 0x0001040507060302ull,
     0x0204050706030100ull, 0x0002040507060301ull, 0x0102040507060300ull,
     0x0001020405070603ull, 0x0304050706020100ull, 0x0003040507060201ull,
     0x0103040507060200ull, 0x0001030405070602ull, 0x0203040507060100ull,
     0x0002030405070601ull, 0x0102030405070600ull, 0x0001020304050706ull,
     0x0607050403020100ull, 0x0006070504030201ull, 0x0106070504030200ull,
     0x0001060705040302ull, 0x0206070504030100ull, 0x0002060705040301ull,
     0x0102060705040300ull, 0x0001020607050403ull, 0x0306070504020100ull,
     0x0003060705040201ull, 0x0103060705040200ull, 0x0001030607050402ull,
     0x0203060705040100ull, 0x0002030607050401ull, 0x0102030607050400ull,
     0x0001020306070504ull, 0x0406070503020100ull, 0x0004060705030201ull,
     0x0104060705030200ull, 0x0001040607050302ull, 0x0204060705030100ull,
     0x0002040607050301ull, 0x0102040607050300ull, 0x0001020406070503ull,
     0x0304060705020100ull, 0x0003040607050201ull, 0x0103040607050200ull,
     0x0001030406070502ull, 0x0203040607050100ull, 0x0002030406070501ull,
     0x0102030406070500ull, 0x0001020304060705ull, 0x0506070403020100ull,
     0x0005060704030201ull, 0x0105060704030200ull, 0x0001050607040302ull,
     0x0205060704030100ull, 0x0002050607040301ull, 0x0102050607040300ull,
     0x0001020506070403ull, 0x0305060704020100ull, 0x0003050607040201ull,
     0x0103050607040200ull, 0x0001030506070402ull, 0x0203050607040100ull,
     0x0002030506070401ull, 0x0102030506070400ull, 0x0001020305060704ull,
     0x0405060703020100ull, 0x0004050607030201ull, 0x0104050607030200ull,
     0x0001040506070302ull, 0x0204050607030100ull, 0x0002040506070301ull,
     0x0102040506070300ull, 0x0001020405060703ull, 0x0304050607020100ull,
     0x0003040506070201ull, 0x0103040506070200ull, 0x0001030405060702ull,
     0x0203040506070100ull, 0x0002030405060701ull, 0x0102030405060700ull,
     0x0001020304050607ull, 0x0706050403020100ull, 0x0007060504030201ull,
     0x0107060504030200ull, 0x0001070605040302ull, 0x0207060504030100ull,
     0x0002070605040301ull, 0x0102070605040300ull, 0x0001020706050403ull,
     0x0307060504020100ull, 0x0003070605040201ull, 0x0103070605040200ull,
     0x0001030706050402ull, 0x0203070605040100ull, 0x0002030706050401ull,
     0x0102030706050400ull, 0x0001020307060504ull, 0x0407060503020100ull,
     0x0004070605030201ull, 0x0104070605030200ull, 0x0001040706050302ull,
     0x0204070605030100ull, 0x0002040706050301ull, 0x0102040706050300ull,
     0x0001020407060503ull, 0x0304070605020100ull, 0x0003040706050201ull,
     0x0103040706050200ull, 0x0001030407060502ull, 0x0203040706050100ull,
     0x0002030407060501ull, 0x0102030407060500ull, 0x0001020304070605ull,
     0x0507060403020100ull, 0x0005070604030201ull, 0x0105070604030200ull,
     0x0001050706040302ull, 0x0205070604030100ull, 0x0002050706040301ull,
     0x0102050706040300ull, 0x0001020507060403ull, 0x0305070604020100ull,
     0x0003050706040201ull, 0x0103050706040200ull, 0x0001030507060402ull,
     0x0203050706040100ull, 0x0002030507060401ull, 0x0102030507060400ull,
     0x0001020305070604ull, 0x0405070603020100ull, 0x0004050706030201ull,
     0x0104050706030200ull, 0x0001040507060302ull, 0x0204050706030100ull,
     0x0002040507060301ull, 0x0102040507060300ull, 0x0001020405070603ull,
     0x0304050706020100ull, 0x0003040507060201ull, 0x0103040507060200ull,
     0x0001030405070602ull, 0x0203040507060100ull, 0x0002030405070601ull,
     0x0102030405070600ull, 0x0001020304050706ull, 0x0607050403020100ull,
     0x0006070504030201ull, 0x0106070504030200ull, 0x0001060705040302ull,
     0x0206070504030100ull, 0x0002060705040301ull, 0x0102060705040300ull,
     0x0001020607050403ull, 0x0306070504020100ull, 0x0003060705040201ull,
     0x0103060705040200ull, 0x0001030607050402ull, 0x0203060705040100ull,
     0x0002030607050401ull, 0x0102030607050400ull, 0x0001020306070504ull,
     0x0406070503020100ull, 0x0004060705030201ull, 0x0104060705030200ull,
     0x0001040607050302ull, 0x0204060705030100ull, 0x0002040607050301ull,
     0x0102040607050300ull, 0x0001020406070503ull, 0x0304060705020100ull,
     0x0003040607050201ull, 0x0103040607050200ull, 0x0001030406070502ull,
     0x0203040607050100ull, 0x0002030406070501ull, 0x0102030406070500ull,
     0x0001020304060705ull, 0x0506070403020100ull, 0x0005060704030201ull,
     0x0105060704030200ull, 0x0001050607040302ull, 0x0205060704030100ull,
     0x0002050607040301ull, 0x0102050607040300ull, 0x0001020506070403ull,
     0x0305060704020100ull, 0x0003050607040201ull, 0x0103050607040200ull,
     0x0001030506070402ull, 0x0203050607040100ull, 0x0002030506070401ull,
     0x0102030506070400ull, 0x0001020305060704ull, 0x0405060703020100ull,
     0x0004050607030201ull, 0x0104050607030200ull, 0x0001040506070302ull,
     0x0204050607030100ull, 0x0002040506070301ull, 0x0102040506070300ull,
     0x0001020405060703ull, 0x0304050607020100ull, 0x0003040506070201ull,
     0x0103040506070200ull, 0x0001030405060702ull, 0x0203040506070100ull,
     0x0002030405060701ull, 0x0102030405060700ull, 0x0001020304050607ull};

// vectorized Bloom filter probe (single direction)
size_t bloom_simd_single(int32_t *keys, int32_t *vals, size_t tuples, int32_t *filter,
                         int32_t factors[8], size_t functions, uint8_t log_filter_size,
                         int32_t *keys_out, int32_t *vals_out)
{
	size_t j, m, i = 0, o = 0, t = tuples - 8;
	__m256i facts = _mm256_loadu_si256((__m256i*) factors);
	__m128i shift = _mm_cvtsi32_si128(32 - log_filter_size);
	__m256i mask_k = _mm256_set1_epi32(functions);
	__m256i mask_31 = _mm256_set1_epi32(31);
	__m256i mask_1 = _mm256_set1_epi32(1);
	__m256i mask_0 = _mm256_set1_epi32(0);
	// non-constant registers
	__m256i key, val, fun, inv = _mm256_cmpeq_epi32(mask_1, mask_1);
	if (tuples >= 8) do {
		// load new items (mask_0 out reloads)
		__m256i new_key = _mm256_maskload_epi32(&keys[i], inv);
		__m256i new_val = _mm256_maskload_epi32(&vals[i], inv);
		// reset old items
		fun = _mm256_andnot_si256(inv, fun);
		key = _mm256_andnot_si256(inv, key);
		val = _mm256_andnot_si256(inv, val);
		// pick hash function
		__m256i fact = _mm256_permutevar8x32_epi32(facts, fun);
		fun = _mm256_add_epi32(fun, mask_1);
		// combine old with new items
		key = _mm256_or_si256(key, new_key);
		val = _mm256_or_si256(val, new_val);
		// hash the keys and check who is almost dmask_1
		__m256i hash = _mm256_mullo_epi32(key, fact);
		__m256i last = _mm256_cmpeq_epi32(fun, mask_k);
		hash = _mm256_srl_epi32(hash, shift);
		// check bitmap
		__m256i div_32 = _mm256_srli_epi32(hash, 5);
		div_32 = _mm256_i32gather_epi32(filter, div_32, 4);
		__m256i mod_32 = _mm256_and_si256(hash, mask_31);
		mod_32 = _mm256_sllv_epi32(mask_1, mod_32);
		div_32 = _mm256_and_si256(div_32, mod_32);
		inv = _mm256_cmpeq_epi32(div_32, mask_0);
		// branch to print winners
		if (!_mm256_testz_si256(div_32, last)) {
			__m256i mask = _mm256_andnot_si256(inv, last);
			m = _mm256_movemask_ps(_mm256_castsi256_ps(mask));
			__m128i perm_half = _mm_loadl_epi64((__m128i*) &perm[m ^ 255]);
			__m256i perm = _mm256_cvtepi8_epi32(perm_half);
			mask = _mm256_permutevar8x32_epi32(mask, perm);
			__m256i key_out = _mm256_permutevar8x32_epi32(key, perm);
			__m256i val_out = _mm256_permutevar8x32_epi32(val, perm);
			_mm256_maskstore_epi32(&keys_out[o], mask, key_out);
			_mm256_maskstore_epi32(&vals_out[o], mask, val_out);
			o += _mm_popcnt_u64(m);
		}
		// permute to get rid of losers (and winners)
		inv = _mm256_or_si256(inv, last);
		m = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
		__m128i perm_half = _mm_loadl_epi64((__m128i*) &perm[m]);
		__m256i perm = _mm256_cvtepi8_epi32(perm_half);
		inv = _mm256_permutevar8x32_epi32(inv, perm);
		fun = _mm256_permutevar8x32_epi32(fun, perm);
		key = _mm256_permutevar8x32_epi32(key, perm);
		val = _mm256_permutevar8x32_epi32(val, perm);
		i += _mm_popcnt_u64(m);
	} while (i <= t);
	// copy last items
	int32_t l_keys[8];
	int32_t l_vals[8];
	_mm256_storeu_si256((__m256i*) l_keys, key);
	_mm256_storeu_si256((__m256i*) l_vals, val);
	j = _mm256_movemask_ps(_mm256_castsi256_ps(inv));
	j = 8 - _mm_popcnt_u64(j);
	assert(i + j <= tuples);
	for (i += j ; i != tuples ; ++i, ++j) {
		l_keys[j] = keys[i];
		l_vals[j] = vals[i];
	}
	// process remaining items with scalar code
	for (i = 0 ; i != j ; ++i) {
		int32_t key1 = l_keys[i];
		size_t f = 0;
		do {
			uint32_t h = key1 * factors[f++];
			h >>= 32 - log_filter_size;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed]"
			:: "r"(filter), "r"(h)
			: "cc" : failed);
		} while (f != functions);
		vals_out[o] = l_vals[i];
		keys_out[o++] = key1;
failed:;
	}
	return o;
}

// vectorized Bloom filter probe (double direction)
size_t bloom_simd_double(int32_t *keys, int32_t *vals, size_t tuples, int32_t *filter,
                         int32_t factors[8], size_t functions, uint8_t log_filter_size,
                         int32_t *keys_out, int32_t *vals_out)
{
	size_t i, j, m, m_L, m_H, o = 0, i_L = 0, i_H = tuples - 8;
	__m256i facts = _mm256_loadu_si256((__m256i*) factors);
	__m256i reverse = _mm256_set_epi32(0,1,2,3,4,5,6,7);
	__m128i shift = _mm_cvtsi32_si128(32 - log_filter_size);
	__m256i mask_k = _mm256_set1_epi32(functions);
	__m256i mask_31 = _mm256_set1_epi32(31);
	__m256i mask_0 = _mm256_set1_epi32(0);
	__m256i mask_1 = _mm256_set1_epi32(1);
	// non-constant registers
	__m256i key_L, val_L, fun_L, inv_L = _mm256_cmpeq_epi32(mask_1, mask_1);
	__m256i key_H, val_H, fun_H, inv_H = _mm256_cmpeq_epi32(mask_1, mask_1);
	while (i_H >= i_L + 8) {
		// reverse the reading mask of the high-to-low load
		__m256i rev_inv_H = _mm256_permutevar8x32_epi32(inv_H, reverse);
		// load 8 items (some are reloads)
		__m256i new_key_L = _mm256_maskload_epi32(&keys[i_L], inv_L);
		__m256i new_val_L = _mm256_maskload_epi32(&vals[i_L], inv_L);
		__m256i new_key_H = _mm256_maskload_epi32(&keys[i_H], rev_inv_H);
		__m256i new_val_H = _mm256_maskload_epi32(&vals[i_H], rev_inv_H);
		// reset old items
		fun_L = _mm256_andnot_si256(inv_L, fun_L);
		fun_H = _mm256_andnot_si256(inv_H, fun_H);
		key_L = _mm256_andnot_si256(inv_L, key_L);
		key_H = _mm256_andnot_si256(inv_H, key_H);
		val_L = _mm256_andnot_si256(inv_L, val_L);
		val_H = _mm256_andnot_si256(inv_H, val_H);
		// reverse key and value read from end
		new_key_H = _mm256_permutevar8x32_epi32(new_key_H, reverse);
		new_val_H = _mm256_permutevar8x32_epi32(new_val_H, reverse);
		// combine new and old items
		key_L = _mm256_or_si256(key_L, new_key_L);
		val_L = _mm256_or_si256(val_L, new_val_L);
		key_H = _mm256_or_si256(key_H, new_key_H);
		val_H = _mm256_or_si256(val_H, new_val_H);
		// pick hash function
		fun_L = _mm256_andnot_si256(inv_L, fun_L);
		fun_H = _mm256_andnot_si256(inv_H, fun_H);
		__m256i fact_L = _mm256_permutevar8x32_epi32(facts, fun_L);
		__m256i fact_H = _mm256_permutevar8x32_epi32(facts, fun_H);
		fun_L = _mm256_add_epi32(fun_L, mask_1);
		fun_H = _mm256_add_epi32(fun_H, mask_1);
		// hash the keys and check who is almost dmask_1
		__m256i hash_L = _mm256_mullo_epi32(key_L, fact_L);
		__m256i hash_H = _mm256_mullo_epi32(key_H, fact_H);
		__m256i last_L = _mm256_cmpeq_epi32(fun_L, mask_k);
		__m256i last_H = _mm256_cmpeq_epi32(fun_H, mask_k);
		hash_L = _mm256_srl_epi32(hash_L, shift);
		hash_H = _mm256_srl_epi32(hash_H, shift);
		// check bitmap
		__m256i div_32_L = _mm256_srli_epi32(hash_L, 5);
		__m256i div_32_H = _mm256_srli_epi32(hash_H, 5);
		div_32_L = _mm256_i32gather_epi32(filter, div_32_L, 4);
		div_32_H = _mm256_i32gather_epi32(filter, div_32_H, 4);
		__m256i mod_32_L = _mm256_and_si256(hash_L, mask_31);
		__m256i mod_32_H = _mm256_and_si256(hash_H, mask_31);
		mod_32_L = _mm256_sllv_epi32(mask_1, mod_32_L);
		mod_32_H = _mm256_sllv_epi32(mask_1, mod_32_H);
		div_32_L = _mm256_and_si256(div_32_L, mod_32_L);
		div_32_H = _mm256_and_si256(div_32_H, mod_32_H);
		inv_L = _mm256_cmpeq_epi32(div_32_L, mask_0);
		inv_H = _mm256_cmpeq_epi32(div_32_H, mask_0);
		// branch to print low-to-high winners
		if (!_mm256_testz_si256(div_32_L, last_L)) {
			__m256i mask = _mm256_andnot_si256(inv_L, last_L);
			m = _mm256_movemask_ps(_mm256_castsi256_ps(mask));
			__m128i perm_half = _mm_loadl_epi64((__m128i*) &perm[m ^ 255]);
			__m256i perm = _mm256_cvtepi8_epi32(perm_half);
			mask = _mm256_permutevar8x32_epi32(mask, perm);
			__m256i key_out = _mm256_permutevar8x32_epi32(key_L, perm);
			__m256i val_out = _mm256_permutevar8x32_epi32(val_L, perm);
			_mm256_maskstore_epi32(&keys_out[o], mask, key_out);
			_mm256_maskstore_epi32(&vals_out[o], mask, val_out);
			o += _mm_popcnt_u64(m);
		}
		// branch to print high-to-low winners
		if (!_mm256_testz_si256(div_32_H, last_H)) {
			__m256i mask = _mm256_andnot_si256(inv_H, last_H);
			m = _mm256_movemask_ps(_mm256_castsi256_ps(mask));
			__m128i perm_half = _mm_loadl_epi64((__m128i*) &perm[m ^ 255]);
			__m256i perm = _mm256_cvtepi8_epi32(perm_half);
			mask = _mm256_permutevar8x32_epi32(mask, perm);
			__m256i key_out = _mm256_permutevar8x32_epi32(key_H, perm);
			__m256i val_out = _mm256_permutevar8x32_epi32(val_H, perm);
			_mm256_maskstore_epi32(&keys_out[o], mask, key_out);
			_mm256_maskstore_epi32(&vals_out[o], mask, val_out);
			o += _mm_popcnt_u64(m);
		}
		// permute to get rid of losers (and winners)
		inv_L = _mm256_or_si256(inv_L, last_L);
		inv_H = _mm256_or_si256(inv_H, last_H);
		m_L = _mm256_movemask_ps(_mm256_castsi256_ps(inv_L));
		m_H = _mm256_movemask_ps(_mm256_castsi256_ps(inv_H));
		__m128i perm_half_L = _mm_loadl_epi64((__m128i*) &perm[m_L]);
		__m128i perm_half_H = _mm_loadl_epi64((__m128i*) &perm[m_H]);
		__m256i perm_L = _mm256_cvtepi8_epi32(perm_half_L);
		__m256i perm_H = _mm256_cvtepi8_epi32(perm_half_H);
		inv_L = _mm256_permutevar8x32_epi32(inv_L, perm_L);
		inv_H = _mm256_permutevar8x32_epi32(inv_H, perm_H);
		fun_L = _mm256_permutevar8x32_epi32(fun_L, perm_L);
		fun_H = _mm256_permutevar8x32_epi32(fun_H, perm_H);
		key_L = _mm256_permutevar8x32_epi32(key_L, perm_L);
		key_H = _mm256_permutevar8x32_epi32(key_H, perm_H);
		val_L = _mm256_permutevar8x32_epi32(val_L, perm_L);
		val_H = _mm256_permutevar8x32_epi32(val_H, perm_H);
		i_L += _mm_popcnt_u64(m_L);
		i_H -= _mm_popcnt_u64(m_H);
	}
	// last keys, values, mask_31ets
	int32_t l_keys[16];
	int32_t l_vals[16];
	_mm256_storeu_si256((__m256i*) &l_keys[0], key_L);
	_mm256_storeu_si256((__m256i*) &l_vals[0], val_L);
	j = _mm256_movemask_ps(_mm256_castsi256_ps(inv_L));
	j = 8 - _mm_popcnt_u64(j);
	_mm256_storeu_si256((__m256i*) &l_keys[j], key_H);
	_mm256_storeu_si256((__m256i*) &l_vals[j], val_H);
	i = _mm256_movemask_ps(_mm256_castsi256_ps(inv_H));
	i = 8 - _mm_popcnt_u64(i);
	// copy unread items
	i_L += j;
	i_H += 8 - i;
	j += i;
	for (; i_L != i_H ; ++i_L, ++j) {
		l_keys[j] = keys[i_L];
		l_vals[j] = vals[i_L];
	}
	// process remaining items with scalar code
	for (i = 0 ; i != j ; ++i) {
		int32_t key1 = l_keys[i];
		size_t f = 0;
		do {
			uint32_t h = key1 * factors[f++];
			h >>= 32 - log_filter_size;
			asm goto("btl	%1, (%0)\n\t"
			         "jnc	%l[failed]"
			:: "r"(filter), "r"(h)
			: "cc" : failed);
		} while (f != functions);
		vals_out[o] = l_vals[i];
		keys_out[o++] = key1;
failed:;
	}
	return o;
}

// thread information
typedef struct {
	pthread_t id;                // POSIX thread id
	int threads;                 // number of threads (T)
	int thread_id;               // thread number in [0,T-1]
	uint32_t seed;               // random generator seed
	uint8_t log_filter_size;     // log (Bloom filter bits)
	int32_t *factors;            // multiplicative factors
	uint64_t times[4];           // time per method
	uint64_t cycles[4];          // cycles per method
	double filter_rate;          // rate of qualifying tuples
	size_t errors;               // number of false positives
	size_t passed;               // number of tuples that qualified
	size_t functions;            // number of Bloom filter functions
	size_t outer_tuples;         // number of outer table tuples
	size_t inner_tuples;         // number of inner table tuples
	volatile int32_t *filter;    // Bloom filter
	volatile int32_t *unique;    // inner side tuples
	pthread_barrier_t *barriers; // synchronization barriers
} info_t;

// thread function
void *run(void *arg)
{
	info_t *d = (info_t*) arg;
	assert(pthread_equal(pthread_self(), d->id));
	bind_thread(d->thread_id);
	size_t i, r, e, f, m;
	// thread local cardinalities
	size_t local_inner_tuples = d->inner_tuples / d->threads;
	size_t local_outer_tuples = d->outer_tuples / d->threads;
	if (d->thread_id == 0) {
		local_inner_tuples += d->inner_tuples % d->threads;
		local_outer_tuples += d->outer_tuples % d->threads;
	}
	// tables in volatile pointers for safe read-write access
	volatile int32_t *vol_unique = d->unique;
	volatile int32_t *vol_filter = d->filter;
	// tables in non-volatile pointers for read-only access
	int32_t *unique = (int32_t*) d->unique;
	int32_t *filter = (int32_t*) d->filter;
	// initialize random generator
	mt32_t *gen = mt32_init(d->seed);
	int32_t payload_factor = mt32_next(gen) | 1;
	int32_t ordered_factor = mt32_next(gen) | 3;
	// configure the bloom filter based on given error rate
	size_t functions = d->functions;
	int32_t *factors = d->factors;
	uint8_t log_filter_size = d->log_filter_size;
	uint8_t shift = 32 - d->log_filter_size;
	// initialize inner table (built) => unique keys only
	size_t buckets_of_unique = d->inner_tuples / 0.7;
	const int32_t knuth_factor = 0x9e3779b1;
	for (i = 0 ; i != local_inner_tuples ; ++i) {
		int32_t key, tab;
		// generate new unique key
		do {
			key = mt32_next(gen);
			uint64_t h = (uint32_t) (key * knuth_factor);
			h = (h * buckets_of_unique) >> 32;
			while ((tab = vol_unique[h]) != 0 && tab != key)
				if (++h == buckets_of_unique) h = 0;
			// atomic update in table
			asm("lock\n\t"
			    "cmpxchgl	%2, (%1)"
			: "=a"(tab)
			: "r"(&vol_unique[h]), "r"(key), "a"(tab)
			: "cc", "memory");
		} while (tab == key);
		// add key in the Bloom filter
		for (f = 0 ; f != functions ; ++f) {
			uint32_t h = key * factors[f];
			h >>= shift;
			// atomic bit set in the Bloom filter
			asm("lock\n\t"
			    "btsl	%1, (%0)"
			:: "r"(vol_filter), "r"(h)
			: "cc", "memory");
		}
	}
	// wait until all threads finish generating the right side keys
	pthread_barrier_wait(&d->barriers[0]);
	// initialize outer table (probed) => keys & payloads
	int32_t *keys = malloc(local_outer_tuples * sizeof(uint32_t));
	int32_t *vals = malloc(local_outer_tuples * sizeof(uint32_t));
	size_t qualify_correctly = 0;
	uint32_t error_limit = ~0;
	error_limit *= d->filter_rate;
	for (i = r = 0 ; i != local_outer_tuples ; ++i) {
		int32_t key, tab;
		uint64_t h;
		// should it qualify or not ?
		if (mt32_next(gen) > error_limit)
			// create an item that is not in the hash table
			do {
				key = mt32_next(gen);
				h = (uint32_t) (key * knuth_factor);
				h = (h * buckets_of_unique) >> 32;
				while ((tab = unique[h]) != 0 && tab != key)
					if (++h == buckets_of_unique) h = 0;
			} while (key == tab);
		else {
			// create an item that is in the hash table
			do {
				h = mt32_next(gen);
				h = (h * buckets_of_unique) >> 32;
			} while ((key = unique[h]) == 0);
			qualify_correctly++;
		}
		// generate a matching payload per key
		keys[i] = key;
		vals[i] = key * payload_factor;
	}
	free(gen);
	// initialize output
	int32_t *keys_out = malloc(local_outer_tuples * sizeof(int32_t));
	int32_t *vals_out = malloc(local_outer_tuples * sizeof(int32_t));
	for (i = 0 ; i != local_outer_tuples ; ++i) {
		keys_out[i] = 0xDEADBEEF;
		vals_out[i] = 0xCAFEBABE;
	}
	// measure all implementations (same prototype)
	size_t (*method[4]) (int32_t*, int32_t*, size_t, int32_t*,
	                     int32_t*, size_t, uint8_t, int32_t*, int32_t*) =
	{bloom_scalar_soft, bloom_scalar_hard, bloom_simd_single, bloom_simd_double};
	int32_t checksum, ordered_checksum;
	for (m = 0 ; m != 4 ; ++m) {
		// sync all threads
		pthread_barrier_wait(&d->barriers[m + m + 1]);
		// run method with thread local timing
		uint64_t t = thread_time();
		uint64_t tsc_begin = _rdtsc();
		const char* env = getenv("REPEAT_CNT");
		const uint64_t repeat_cnt = atoi(env);
		for (uint64_t rep = 0; rep < repeat_cnt; rep++) {
			r = method[m](keys, vals, local_outer_tuples, filter, factors,
										functions, log_filter_size, keys_out, vals_out);
		}
		uint64_t tsc_end = _rdtsc();
		t = thread_time() - t;
		// sync all threads again
		pthread_barrier_wait(&d->barriers[m + m + 2]);
		// save and check results
		d->times[m] = t / repeat_cnt;
		d->cycles[m] = (tsc_end - tsc_begin) / repeat_cnt;
		if (m) assert(r == d->passed);
		int32_t s = 0, o = 0;
		for (i = e = 0 ; i != r ; ++i) {
			// check if key qualifies the filter
			int32_t key = keys_out[i];
			for (f = 0 ; f != functions ; ++f) {
				uint64_t h = (uint32_t) (key * factors[f]);
				h >>= shift;
				assert(((filter[h >> 5]) & (1 << (h & 31))) != 0);
			}
			// check if false positive or not
			uint64_t h = (uint32_t) (key * knuth_factor);
			h = (h * buckets_of_unique) >> 32;
			while (unique[h] != 0 && unique[h] != key)
				if (++h == buckets_of_unique) h = 0;
			e += unique[h] == 0 ? 1 : 0;
			// check matching payload and update checksums
			assert(vals_out[i] == key * payload_factor);
			s += key;
			o += key + o * ordered_factor;
		}
		if (m) {
			assert(e == d->errors);
			assert(s == checksum);
			assert(m == 3 || o == ordered_checksum);
		} else {
			d->errors = e;
			d->passed = r;
			checksum = s;
			ordered_checksum = o;
			assert(r - e == qualify_correctly);
		}
	}
	// thread cleanup and exit
	free(keys);
	free(vals);
	free(keys_out);
	free(vals_out);
	pthread_exit(NULL);
}

int is_power_of_two(uint64_t x)
{
	return x != 0 && (x & (x - 1)) == 0;
}

int thousands_letter(uint8_t x)
{
	return x < 10 ? ' ' : x < 20 ? 'K' : x < 30 ? 'M' : x < 40 ? 'G' : '?';
}

int power_of_two_divisor(int32_t x)
{
	int p = 0;
	while (((1 << p) & x) == 0 && p != 32) p++;
	return p;
}

uint8_t parse_power_of_two(const char *bloom_size_string)
{
	assert(isdigit(*bloom_size_string));
	uint64_t bloom_size = 0;
	do {
		bloom_size = (bloom_size * 10) + *bloom_size_string++ - '0';
	} while (isdigit(*bloom_size_string));
	if (*bloom_size_string) {
		assert(*bloom_size_string == 'K' ||
		       *bloom_size_string == 'M' ||
		       *bloom_size_string == 'G');
		if      (*bloom_size_string == 'K') bloom_size <<= 10;
		else if (*bloom_size_string == 'M') bloom_size <<= 20;
		else if (*bloom_size_string == 'G') bloom_size <<= 30;
		bloom_size_string++;
	}
	assert(*bloom_size_string == 0);
	assert(is_power_of_two(bloom_size));
	uint8_t log_bloom_size = 1;
	while ((((uint64_t) 1) << log_bloom_size) != bloom_size)
		log_bloom_size++;
	return log_bloom_size;
}

int main(int argc, char **argv)
{
	int i, j, t;
	// you must give 5 or 6 arguments
	if (argc < 6 || argc > 7) {
		fprintf(stderr, "Usage: %s outer_tuples inner_tuples ", argv[0]);
		fprintf(stderr, "filter_size hash_functions filter_rate [threads]\n");
		return EXIT_FAILURE;
	}
	// read and check input parameters
	size_t outer_tuples = atoll(argv[1]);
	size_t inner_tuples = atoll(argv[2]);
	uint8_t log_filter_size = parse_power_of_two(argv[3]) + 3;
	int hash_functions = atoi(argv[4]);
	double filter_rate = atof(argv[5]);
	int threads = argc > 6 ? atoi(argv[6]) : hardware_threads();
	// check arguments
	assert(filter_rate >= 0.0 && filter_rate <= 1.0);
	assert(hash_functions >= 1 && hash_functions <= 6);
	assert(threads > 0 && threads <= hardware_threads());
	// expected error using approximation formula
	const size_t one = 1;
	double bits_per_item = (one << log_filter_size) * 1.0 / inner_tuples;
	double error = pow(1 - exp(hash_functions * -1.0 / bits_per_item), hash_functions);
	// print configuration
	fprintf(stderr, "Outer table tuples: %ld\n", outer_tuples);
	fprintf(stderr, "Inner table tuples: %ld\n", inner_tuples);
	fprintf(stderr, "Bloom filter size: %ld %cB\n", one << ((log_filter_size - 3) % 10),
	                                                thousands_letter(log_filter_size - 3));
	fprintf(stderr, "Bits / item: %.2f\n", (one << log_filter_size) * 1.0 / inner_tuples);
	fprintf(stderr, "Hash functions: %d\n", hash_functions);
	fprintf(stderr, "Threads: %d / %d\n", threads, hardware_threads());
	fprintf(stderr, "Expected filter rate: %.5f%%\n", filter_rate * 100.0);
	fprintf(stderr, "Expected error  rate: %.5f%%\n", error * 100.0);
	// generate distinct primes as multiplicative hashing factors
	srand(time(NULL));
	int32_t factors[8];
	for (i = 0 ; i != hash_functions ; ++i) {
		int32_t f = rand();
		f += f + 1;
		// ensure pairwise differences are a low power of 2
		for (j = 0 ; j != i ; ++j)
			if (power_of_two_divisor(factors[j] - f) > 3)
				break;
		if (i != j) i--;
		else factors[i] = f;
	}
	// allocate bloom filter and table for distinct keys
	size_t distinct_buckets_of_unique = inner_tuples / 0.7;
	size_t filter_words = one << (log_filter_size - 5);
	int32_t *unique = calloc(distinct_buckets_of_unique, sizeof(int32_t));
	int32_t *filter = calloc(filter_words, sizeof(int32_t));
	// initialize synchronization barriers
	pthread_barrier_t barriers[9];
	for (t = 0 ; t != 9 ; ++t)
		pthread_barrier_init(&barriers[t], NULL, threads);
	// initialize threads
	info_t info[threads];
	for (t = 0 ; t != threads ; ++t) {
		info[t].inner_tuples = inner_tuples;
		info[t].outer_tuples = outer_tuples;
		info[t].filter_rate = filter_rate;
		info[t].log_filter_size = log_filter_size;
		info[t].functions = hash_functions;
		info[t].factors = factors;
		info[t].filter = filter;
		info[t].unique = unique;
		info[t].seed = rand();
		info[t].threads = threads;
		info[t].thread_id = t;
		info[t].barriers = barriers;
		pthread_create(&info[t].id, NULL, run, (void*) &info[t]);
	}
	// wait for threads to finish and gather timing results
	uint64_t errors = 0, passed = 0, times[4] = {0, 0, 0, 0}, cycles[4] = {0, 0, 0, 0};
	for (t = 0 ; t != threads ; ++t) {
		pthread_join(info[t].id, NULL);
		for (i = 0 ; i != 4 ; ++i) {
			times[i] += info[t].times[i];
			cycles[i] += info[t].cycles[i];
		}
		errors += info[t].errors;
		passed += info[t].passed;
	}

	// print results
	fprintf(stderr, "Measured filter rate: %.5f%%\n", (passed - errors) * 100.0 / outer_tuples);
	fprintf(stderr, "Measured error  rate: %.5f%%\n", errors * 100.0 / (outer_tuples - (passed - errors)));
	fprintf(stderr, "Scalar soft: %7.2f mtps\n", (outer_tuples * 1000.0) / ((times[0] * 1.0) / threads));
	fprintf(stderr, "Scalar hard: %7.2f mtps\n", (outer_tuples * 1000.0) / ((times[1] * 1.0) / threads));
	fprintf(stderr, "SIMD single: %7.2f mtps\n", (outer_tuples * 1000.0) / ((times[2] * 1.0) / threads));
	fprintf(stderr, "SIMD double: %7.2f mtps\n", (outer_tuples * 1000.0) / ((times[3] * 1.0) / threads));
	fprintf(stderr, "Scalar soft: %7.6f cycles/outer tuple\n", (cycles[0] * 1.0 / outer_tuples) / threads);
	fprintf(stderr, "Scalar hard: %7.6f cycles/outer tuple\n", (cycles[1] * 1.0 / outer_tuples) / threads);
	fprintf(stderr, "SIMD single: %7.6f cycles/outer tuple\n", (cycles[2] * 1.0 / outer_tuples) / threads);
	fprintf(stderr, "SIMD double: %7.6f cycles/outer tuple\n", (cycles[3] * 1.0 / outer_tuples) / threads);
	// cleanup and exit
	for (t = 0 ; t != 9 ; ++t)
		pthread_barrier_destroy(&barriers[t]);
	free(filter);
	free(unique);
	return EXIT_SUCCESS;
}
