// SPDX-License-Identifier: 0BSD

///////////////////////////////////////////////////////////////////////////////
//
/// \file       crc32_table.c
/// \brief      Precalculated CRC32 table with correct endianness
//
//  Author:     Lasse Collin
//
///////////////////////////////////////////////////////////////////////////////

#include "crc_common.h"

#if !defined(HAVE_ENCODERS) && defined(NO_CRC32_TABLE)
// No table needed. Use a typedef to avoid an empty translation unit.
typedef void lzma_crc32_dummy;

#else
// Having the declaration here silences clang -Wmissing-variable-declarations.
extern const uint32_t lzma_crc32_table[8][256];

#	ifdef WORDS_BIGENDIAN
#		include "crc32_table_be.h"
#	else
#		include "crc32_table_le.h"
#	endif
#endif
