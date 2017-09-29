/*
 * Derived from the vectorized Bloom filter implementation of
 * Orestis Polychroniou, Department of Computer Science, Columbia University.
 *
 * See file 'vectorized_bloomfilter_columbia.c' for details.
 */
#include "vbf.hpp"

#if defined(__AVX2__)

#include <dtl/filter/blocked_bloomfilter/hash_family.hpp>

namespace columbia {

using hasher = dtl::hash::dyn::mul32;

static constexpr u32 primes[17] {
    596572387u,   // Peter 1
    370248451u,   // Peter 2
    2654435769u,  // Knuth 1
    1799596469u,  // Knuth 2
    0x9e3779b1u,  // https://lowrey.me/exploring-knuths-multiplicative-hash-2/
    2284105051u,  // Impala 3
    1203114875u,  // Impala 1 (odd, not prime)
    1150766481u,  // Impala 2 (odd, not prime)
    2729912477u,  // Impala 4 (odd, not prime)
    1884591559u,  // Impala 5 (odd, not prime)
    770785867u,   // Impala 6 (odd, not prime)
    2667333959u,  // Impala 7 (odd, not prime)
    1550580529u,  // Impala 8 (odd, not prime)
    0xcc9e2d51u,  // Murmur 3 (x86_32 c1)
    0x1b873593u,  // Murmur 3 (x86_32 c2)
    0x85ebca6bu,  // Murmur 3 (finalization mix constant)
    0xc2b2ae35u,  // Murmur 3 (finalization mix constant)
};


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


vbf::vbf(vbf&&) = default;
vbf::~vbf() = default;
//vbf& vbf::operator=(vbf&&) = default;

void
vbf::insert(vbf::word_t* __restrict filter_data, u32 key) {
  // add key in the Bloom filter
  for ($u32 f = 0 ; f != k ; ++f) {
    uint32_t h = hasher::hash(key, f);
    h >>= shift;
    // atomic bit set in the Bloom filter
    asm("lock\n\t"
        "btsl	%1, (%0)"
    :: "r"(filter_data), "r"(h)
    : "cc", "memory");
  }
}

void
vbf::batch_insert(vbf::word_t* __restrict filter_data, u32* keys, u32 key_cnt) {
  for (std::size_t i = 0; i < key_cnt; i++) {
    insert(filter_data, keys[i]);
  }
}

$u1
vbf::contains(const vbf::word_t* __restrict filter_data, u32 key) const {
  $u1 found = false;
  $u32 f = 0;
  do {
    u32 h = hasher::hash(key,f++) >> shift;
    asm goto("btl	%1, (%0)\n\t"
             "jnc	%l[failed]"
             :: "r"(filter_data), "r"(h)
             : "cc" : failed);
  } while (f != k);
  found = true;
failed:
  return found;
}

$u64
vbf::batch_contains(const vbf::word_t* __restrict filter_data,
                    u32* keys, u32 key_cnt, $u32* match_positions, u32 match_offset) const {
  size_t i, j, m, m_L, m_H, o = 0, i_L = 0, i_H = key_cnt - 8;
  __m256i facts = _mm256_loadu_si256((__m256i*) primes);
  __m256i reverse = _mm256_set_epi32(0,1,2,3,4,5,6,7);
  const __m256i seq = _mm256_set_epi32(7,6,5,4,3,2,1,0); // HL: sequence
  __m128i shift = _mm_cvtsi32_si128(this->shift);
  __m256i mask_k = _mm256_set1_epi32(k);
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
    __m256i new_key_L = _mm256_maskload_epi32(reinterpret_cast<i32*>(&keys[i_L]), inv_L);
//    __m256i new_val_L = _mm256_maskload_epi32(&vals[i_L], inv_L);
    __m256i new_val_L = _mm256_add_epi32(_mm256_set1_epi32(i_L), seq); // HL: carry tuple IDs instead of values
    __m256i new_key_H = _mm256_maskload_epi32(reinterpret_cast<i32*>(&keys[i_H]), rev_inv_H);
//    __m256i new_val_H = _mm256_maskload_epi32(&vals[i_H], rev_inv_H);
    __m256i new_val_H = _mm256_add_epi32(_mm256_set1_epi32(i_H), seq); // HL: carry tuple IDs instead of values
    // reset old items
    fun_L = _mm256_andnot_si256(inv_L, fun_L);
    fun_H = _mm256_andnot_si256(inv_H, fun_H);
    key_L = _mm256_andnot_si256(inv_L, key_L);
    key_H = _mm256_andnot_si256(inv_H, key_H);
    val_L = _mm256_andnot_si256(inv_L, val_L);
    val_H = _mm256_andnot_si256(inv_H, val_H);
    // reverse key and value read from end
    new_key_H = _mm256_permutevar8x32_epi32(new_key_H, reverse);
//    new_val_H = _mm256_permutevar8x32_epi32(new_val_H, reverse);
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
    div_32_L = _mm256_i32gather_epi32(reinterpret_cast<i32*>(filter_data), div_32_L, 4);
    div_32_H = _mm256_i32gather_epi32(reinterpret_cast<i32*>(filter_data), div_32_H, 4);
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
//      _mm256_maskstore_epi32(&keys_out[o], mask, key_out);
//      _mm256_maskstore_epi32(&vals_out[o], mask, val_out);
      _mm256_maskstore_epi32(reinterpret_cast<$i32*>(&match_positions[o]), mask, val_out); // HL: store only TIDs
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
//      _mm256_maskstore_epi32(&keys_out[o], mask, key_out);
//      _mm256_maskstore_epi32(&vals_out[o], mask, val_out);
      _mm256_maskstore_epi32(reinterpret_cast<$i32*>(&match_positions[o]), mask, val_out); // HL: store only TIDs
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
    l_vals[j] = i_L; // HL: carry tuple IDs instead of values
  }
  // process remaining items with scalar code
  for (i = 0 ; i != j ; ++i) {
    int32_t key1 = l_keys[i];
    size_t f = 0;
    do {
      uint32_t h = key1 * primes[f++];
      h >>= this->shift;
      asm goto("btl	%1, (%0)\n\t"
      "jnc	%l[failed]"
      :: "r"(filter_data), "r"(h)
      : "cc" : failed);
    } while (f != k);
//    vals_out[o] = l_vals[i];
//    keys_out[o++] = key1;
    // HL: store only TIDs
    match_positions[o] = l_vals[i];
    o++;
    failed:;
  }
  return o;
}

std::string
vbf::name() const {
  return "{\"name\":\"columbia_bloom\",\"size\":" + std::to_string(size_in_bytes())
         + ",\"k\":" + std::to_string(k)
         + "}";
}

std::size_t
vbf::size_in_bytes() const {
  return m / 8;
}

std::size_t
vbf::size() const {
  return vbf::size_in_bytes() / sizeof(word_t);
}

} // namespace columbia

#endif //defined(__AVX2__)