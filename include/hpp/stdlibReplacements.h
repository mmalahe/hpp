/** @file stdlibReplacements.h
* @author Michael Malahe
* @brief Header file for implementations of stdlib functions that may be missing.
* @details
*/

#ifndef HPP_STDLIBREPLACEMENTS_H
#define HPP_STDLIBREPLACEMENTS_H

#include <hpp/config.h>

/**
 * @brief The stdlib function aligned_alloc
 * @param size
 * @param align
 * @details From the llvm implementation.
 */
/// @fixme Check how this license propagates or do own implementation.
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