/** @file simdUtils.h
* @author Michael Malahe
* @brief
*/

#ifndef HPP_SIMDUTILS_H
#define HPP_SIMDUTILS_H

#include "immintrin.h"
#include "emmintrin.h"
#include "smmintrin.h"
#include <hpp/config.h>

// Intrinsics defined by the standard, but not implemented in GCC as of 4.9.3
#define _mm256_set_m128i(hi, lo)  _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 1)
#define _mm256_set_m128(hi, lo)  _mm256_insertf128_ps(_mm256_castps128_ps256(lo), (hi), 1)
#define _mm256_loadu2_m128d(hi, lo) _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(lo)), _mm_loadu_pd(hi), 1);

// Convenience defines in this library only
#ifdef HPP_USE_SSE
inline __m128 _mm128_loadu2_m64(float const *hiaddr, float const *loaddr) {
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wuninitialized"
    __m128 val;
    val = _mm_loadl_pi(val, (__m64*)loaddr);
    val = _mm_loadh_pi(val, (__m64*)hiaddr);
    return val;
    #pragma GCC diagnostic pop
}
#endif

#ifdef HPP_USE_AVX
inline __m256 _mm256_loadu4_m64(float const *p3, float const *p2, float const *p1, float const *p0){
    return _mm256_set_m128(_mm128_loadu2_m64(p3, p2), _mm128_loadu2_m64(p1, p0));
};
#endif

#endif /* HPP_SIMDUTILS_H */