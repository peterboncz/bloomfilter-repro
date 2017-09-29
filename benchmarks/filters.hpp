#pragma once

#include "filter/api.hpp"
//#include "filter/blocked_bloom_word.hpp" // deprecated, register blocking is covered by the 'multi-word' implementation
#include "filter/blocked_bloom_multiword.hpp"
#include "filter/blocked_cuckoo.hpp"
#include "filter/bloom.hpp"
#if defined(__AVX2__)
#include "filter/blocked_bloom_impala.hpp"
#include "filter/bloom_columbia.hpp"
#endif
#include "filter/cuckoofilter.hpp"
//#include "filter/dynamic_blocked_bloom.hpp" // deprecated, as the 'multi-word' implementation now supports all configurations
