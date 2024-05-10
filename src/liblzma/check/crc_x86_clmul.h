// SPDX-License-Identifier: 0BSD

///////////////////////////////////////////////////////////////////////////////
//
/// \file       crc_x86_clmul.h
/// \brief      CRC32 and CRC64 implementations using CLMUL instructions.
///
/// The CRC32 and CRC64 implementations use 32/64-bit x86 SSSE3, SSE4.1, and
/// CLMUL instructions. This is compatible with Elbrus 2000 (E2K) too.
///
/// They were derived from
/// https://www.researchgate.net/publication/263424619_Fast_CRC_computation
/// and the public domain code from https://github.com/rawrunprotected/crc
/// (URLs were checked on 2023-10-14).
///
/// While this file has both CRC32 and CRC64 implementations, only one
/// should be built at a time to ensure that crc_simd_body() is inlined
/// even with compilers with which lzma_always_inline expands to plain inline.
/// The version to build is selected by defining BUILDING_CRC32_CLMUL or
/// BUILDING_CRC64_CLMUL before including this file.
///
/// NOTE: Builds for 32-bit x86 use the assembly .S files by default
/// unless configured with --disable-assembler.
//
//  Authors:    Ilya Kurdyukov
//              Hans Jansen
//              Lasse Collin
//              Jia Tan
//
///////////////////////////////////////////////////////////////////////////////

// This file must not be included more than once.
#ifdef LZMA_CRC_X86_CLMUL_H
#	error crc_x86_clmul.h was included twice.
#endif
#define LZMA_CRC_X86_CLMUL_H

#include <immintrin.h>

#if defined(_MSC_VER)
#	include <intrin.h>
#elif defined(HAVE_CPUID_H)
#	include <cpuid.h>
#endif


// EDG-based compilers (Intel's classic compiler and compiler for E2K) can
// define __GNUC__ but the attribute must not be used with them.
// The new Clang-based ICX needs the attribute.
//
// NOTE: Build systems check for this too, keep them in sync with this.
#if (defined(__GNUC__) || defined(__clang__)) && !defined(__EDG__)
#	define crc_attr_target \
		__attribute__((__target__("ssse3,sse4.1,pclmul")))
#else
#	define crc_attr_target
#endif


crc_attr_target
static lzma_always_inline void
crc_simd_body(const uint8_t *buf, const size_t size, __m128i *v0, __m128i *v1,
		const __m128i vfold16, const __m128i initial_crc)
{
	assert(((uintptr_t)buf & 15) == 0);
	assert((size & 15) == 0);

	const __m128i *aligned_buf = (const __m128i *)buf;
	const __m128i *end = (const __m128i*)(buf + size);

	*v0 = _mm_xor_si128(initial_crc, _mm_load_si128(aligned_buf++));

	if (aligned_buf < end) {
		*v1 = _mm_load_si128(aligned_buf++);

		while (aligned_buf < end) {
			*v1 = _mm_xor_si128(*v1, _mm_clmulepi64_si128(
					*v0, vfold16, 0x00));
			*v0 = _mm_xor_si128(*v1, _mm_clmulepi64_si128(
					*v0, vfold16, 0x11));
			*v1 = _mm_load_si128(aligned_buf++);
		}

		*v1 = _mm_xor_si128(*v1, _mm_clmulepi64_si128(
				*v0, vfold16, 0x00));
		*v0 = _mm_xor_si128(*v1, _mm_clmulepi64_si128(
				*v0, vfold16, 0x11));
	}

	*v1 = _mm_srli_si128(*v0, 8);
}


/////////////////////
// x86 CLMUL CRC32 //
/////////////////////

/*
// These functions were used to generate the constants
// at the top of crc32_arch_optimized().
static uint64_t
calc_lo(uint64_t p, uint64_t a, int n)
{
	uint64_t b = 0; int i;
	for (i = 0; i < n; i++) {
		b = b >> 1 | (a & 1) << (n - 1);
		a = (a >> 1) ^ ((0 - (a & 1)) & p);
	}
	return b;
}

// same as ~crc(&a, sizeof(a), ~0)
static uint64_t
calc_hi(uint64_t p, uint64_t a, int n)
{
	int i;
	for (i = 0; i < n; i++)
		a = (a >> 1) ^ ((0 - (a & 1)) & p);
	return a;
}
*/

#ifdef BUILDING_CRC32_CLMUL

crc_attr_target
static uint32_t
crc32_clmul(const uint8_t *buf, size_t size, uint32_t crc)
{
	// uint32_t poly = 0xedb88320;
	const int64_t p = 0x1db710640; // p << 1
	const int64_t mu = 0x1f7011641; // calc_lo(p, p, 32) << 1 | 1
	const int64_t k5 = 0x163cd6124; // calc_hi(p, p, 32) << 1
	const int64_t k4 = 0x0ccaa009e; // calc_hi(p, p, 64) << 1
	const int64_t k3 = 0x1751997d0; // calc_hi(p, p, 128) << 1

	const __m128i vfold4 = _mm_set_epi64x(mu, p);
	const __m128i vfold8 = _mm_set_epi64x(0, k5);
	const __m128i vfold16 = _mm_set_epi64x(k4, k3);

	__m128i v0, v1, v2;

	crc_simd_body(buf, size, &v0, &v1, vfold16,
			_mm_cvtsi32_si128((int32_t)crc));

	v1 = _mm_xor_si128(
			_mm_clmulepi64_si128(v0, vfold16, 0x10), v1); // xxx0
	v2 = _mm_shuffle_epi32(v1, 0xe7); // 0xx0
	v0 = _mm_slli_epi64(v1, 32);  // [0]
	v0 = _mm_clmulepi64_si128(v0, vfold8, 0x00);
	v0 = _mm_xor_si128(v0, v2);   // [1] [2]
	v2 = _mm_clmulepi64_si128(v0, vfold4, 0x10);
	v2 = _mm_clmulepi64_si128(v2, vfold4, 0x00);
	v0 = _mm_xor_si128(v0, v2);   // [2]
	return (uint32_t)_mm_extract_epi32(v0, 2);
}


static uint32_t
crc32_arch_optimized(const uint8_t *buf, size_t size, uint32_t crc)
{
	if (size >= 32) {
		const size_t align16 = (0U - (uintptr_t)buf) & 15;
		if (align16 > 0) {
			crc = crc32_generic(buf, align16, crc);
			buf += align16;
			size -= align16;
		}

		const size_t size16 = size & ~(size_t)15;
		crc = ~crc32_clmul(buf, size16, ~crc);
		buf += size16;
		size &= 15;
	}

	return crc32_generic(buf, size, crc);
}

#endif // BUILDING_CRC32_CLMUL


/////////////////////
// x86 CLMUL CRC64 //
/////////////////////

/*
// These functions were used to generate the constants
// at the top of crc64_arch_optimized().
static uint64_t
calc_lo(uint64_t poly)
{
	uint64_t a = poly;
	uint64_t b = 0;

	for (unsigned i = 0; i < 64; ++i) {
		b = (b >> 1) | (a << 63);
		a = (a >> 1) ^ (a & 1 ? poly : 0);
	}

	return b;
}

static uint64_t
calc_hi(uint64_t poly, uint64_t a)
{
	for (unsigned i = 0; i < 64; ++i)
		a = (a >> 1) ^ (a & 1 ? poly : 0);

	return a;
}
*/

#ifdef BUILDING_CRC64_CLMUL

// MSVC (VS2015 - VS2022) produces bad 32-bit x86 code from the CLMUL CRC
// code when optimizations are enabled (release build). According to the bug
// report, the ebx register is corrupted and the calculated result is wrong.
// Trying to workaround the problem with "__asm mov ebx, ebx" didn't help.
// The following pragma works and performance is still good. x86-64 builds
// and CRC32 CLMUL aren't affected by this problem. The problem does not
// happen in crc_simd_body() either (which is shared with CRC32 CLMUL anyway).
//
// NOTE: Another pragma after crc64_clmul() restores the optimizations.
// If the #if condition here is updated, the other one must be updated too.
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__clang__) \
		&& defined(_M_IX86)
#	pragma optimize("g", off)
#endif

crc_attr_target
static uint64_t
crc64_clmul(const uint8_t *buf, size_t size, uint64_t crc)
{
	// const uint64_t poly = 0xc96c5795d7870f42; // CRC polynomial
	const uint64_t p  = 0x92d8af2baf0e1e85; // (poly << 1) | 1
	const uint64_t mu = 0x9c3e466c172963d5; // (calc_lo(poly) << 1) | 1
	const uint64_t k2 = 0xdabe95afc7875f40; // calc_hi(poly, 1)
	const uint64_t k1 = 0xe05dd497ca393ae4; // calc_hi(poly, k2)

	const __m128i vfold8 = _mm_set_epi64x((int64_t)p, (int64_t)mu);
	const __m128i vfold16 = _mm_set_epi64x((int64_t)k2, (int64_t)k1);

	__m128i v0, v1, v2;

#if defined(__i386__) || defined(_M_IX86)
	crc_simd_body(buf, size, &v0, &v1, vfold16,
			_mm_set_epi64x(0, (int64_t)crc));
#else
	// GCC and Clang would produce good code with _mm_set_epi64x
	// but MSVC needs _mm_cvtsi64_si128 on x86-64.
	crc_simd_body(buf, size, &v0, &v1, vfold16,
			_mm_cvtsi64_si128((int64_t)crc));
#endif

	v1 = _mm_xor_si128(_mm_clmulepi64_si128(v0, vfold16, 0x10), v1);
	v0 = _mm_clmulepi64_si128(v1, vfold8, 0x00);
	v2 = _mm_clmulepi64_si128(v0, vfold8, 0x10);
	v0 = _mm_xor_si128(_mm_xor_si128(v1, _mm_slli_si128(v0, 8)), v2);

#if defined(__i386__) || defined(_M_IX86)
	return ((uint64_t)(uint32_t)_mm_extract_epi32(v0, 3) << 32)
			| (uint32_t)_mm_extract_epi32(v0, 2);
#else
	return (uint64_t)_mm_extract_epi64(v0, 1);
#endif
}

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER) && !defined(__clang__) \
		&& defined(_M_IX86)
#	pragma optimize("", on)
#endif

static uint64_t
crc64_arch_optimized(const uint8_t *buf, size_t size, uint64_t crc)
{
	if (size >= 32) {
		const size_t align16 = (0U - (uintptr_t)buf) & 15;
		if (align16 > 0) {
			crc = crc64_generic(buf, align16, crc);
			buf += align16;
			size -= align16;
		}

		const size_t size16 = size & ~(size_t)15;
		crc = ~crc64_clmul(buf, size16, ~crc);
		buf += size16;
		size &= 15;
	}

	return crc64_generic(buf, size, crc);
}

#endif // BUILDING_CRC64_CLMUL


// Inlining this function duplicates the function body in crc32_resolve() and
// crc64_resolve(), but this is acceptable because this is a tiny function.
static inline bool
is_arch_extension_supported(void)
{
	int success = 1;
	uint32_t r[4]; // eax, ebx, ecx, edx

#if defined(_MSC_VER)
	// This needs <intrin.h> with MSVC. ICC has it as a built-in
	// on all platforms.
	__cpuid(r, 1);
#elif defined(HAVE_CPUID_H)
	// Compared to just using __asm__ to run CPUID, this also checks
	// that CPUID is supported and saves and restores ebx as that is
	// needed with GCC < 5 with position-independent code (PIC).
	success = __get_cpuid(1, &r[0], &r[1], &r[2], &r[3]);
#else
	// Just a fallback that shouldn't be needed.
	__asm__("cpuid\n\t"
			: "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3])
			: "a"(1), "c"(0));
#endif

	// Returns true if these are supported:
	// CLMUL (bit 1 in ecx)
	// SSSE3 (bit 9 in ecx)
	// SSE4.1 (bit 19 in ecx)
	const uint32_t ecx_mask = (1 << 1) | (1 << 9) | (1 << 19);
	return success && (r[2] & ecx_mask) == ecx_mask;

	// Alternative methods that weren't used:
	//   - ICC's _may_i_use_cpu_feature: the other methods should work too.
	//   - GCC >= 6 / Clang / ICX __builtin_cpu_supports("pclmul")
	//
	// CPUID decding is needed with MSVC anyway and older GCC. This keeps
	// the feature checks in the build system simpler too. The nice thing
	// about __builtin_cpu_supports would be that it generates very short
	// code as is it only reads a variable set at startup but a few bytes
	// doesn't matter here.
}
