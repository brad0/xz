// SPDX-License-Identifier: 0BSD

///////////////////////////////////////////////////////////////////////////////
//
/// \file       crc64_table.c
/// \brief      Precalculated CRC64 table with correct endianness
//
//  Author:     Lasse Collin
//
///////////////////////////////////////////////////////////////////////////////

#include "crc_common.h"

#ifdef NO_CRC64_TABLE
// No table needed. Use a typedef to avoid an empty translation unit.
typedef void lzma_crc64_dummy;

#else
// Having the declaration here silences clang -Wmissing-variable-declarations.
extern const uint64_t lzma_crc64_table[4][256];

#	if defined(WORDS_BIGENDIAN)
#		include "crc64_table_be.h"
#	else
#		include "crc64_table_le.h"
#	endif
#endif
