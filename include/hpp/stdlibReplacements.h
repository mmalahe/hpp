/** @file stdlibReplacements.h
* @author Michael Malahe
* @brief Header file for implementations of stdlib functions that may be missing.
* @details
*/

#ifndef HPP_STDLIBREPLACEMENTS_H
#define HPP_STDLIBREPLACEMENTS_H

#include <hpp/config.h>

/**
 * @brief The standard library function aligned_alloc
 * @param size
 * @param align
 * @details This is taken directly from the GCC-4.9 implementation of _mm_malloc,
 * found in gmm_malloc.h. The copyright notice at the top of gmm_malloc.h is
 * reproduced below.
 * 
 * Copyright (C) 2004-2014 Free Software Foundation, Inc.
   This file is part of GCC.
   GCC is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3, or (at your option)
   any later version.
   GCC is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   Under Section 7 of GPL version 3, you are granted additional
   permissions described in the GCC Runtime Library Exception, version
   3.1, as published by the Free Software Foundation.
   You should have received a copy of the GNU General Public License and
   a copy of the GCC Runtime Library Exception along with this program;
   see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
   <http://www.gnu.org/licenses/>.
 */
#ifndef HPP_HAVE_ALIGNED_ALLOC
static __inline__ void* 
aligned_alloc (size_t size, size_t align)
{
  void * malloc_ptr;
  void * aligned_ptr;
  /* Error if align is not a power of two.  */
  if (align & (align - 1))
    {
      errno = EINVAL;
      return ((void*) 0);
    }
  if (size == 0)
    return ((void *) 0);
 /* Assume malloc'd pointer is aligned at least to sizeof (void*).
    If necessary, add another sizeof (void*) to store the value
    returned by malloc. Effectively this enforces a minimum alignment
    of sizeof double. */     
    if (align < 2 * sizeof (void *))
      align = 2 * sizeof (void *);
  malloc_ptr = malloc (size + align);
  if (!malloc_ptr)
    return ((void *) 0);
  /* Align  We have at least sizeof (void *) space below malloc'd ptr. */
  aligned_ptr = (void *) (((size_t) malloc_ptr + align)
			  & ~((size_t) (align) - 1));
  /* Store the original pointer just before p.  */	
  ((void **) aligned_ptr) [-1] = malloc_ptr;
  return aligned_ptr;
}
#endif

#endif /* HPP_STDLIBREPLACEMENTS_H */