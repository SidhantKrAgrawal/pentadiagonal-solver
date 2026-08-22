#ifndef PENTAD_COMMON_H__
#define PENTAD_COMMON_H__

#if FPPREC == 0
#  define FP float
#  define F  f
#elif FPPREC == 1
#  define FP double
#  define F
#endif

#define WARP_SIZE    32
#define ALIGN        32          // 32 byte alignment is required
#define ALIGN_FLOAT  (ALIGN / 4) // 32 byte/ 4bytes/float = 8
#define ALIGN_DOUBLE (ALIGN / 8) // 32 byte/ 8bytes/float = 4
// Maximal dimension that can be used in the library. Defines static arrays
#define MAXDIM          3

#ifdef __cplusplus
#  define EXTERN_C extern "C"
#else
#  define EXTERN_C
#endif

/* This is just a copy of CUSPARSE enums */
typedef enum {
  PENTAD_STATUS_SUCCESS                   = 0,
  PENTAD_STATUS_NOT_INITIALIZED           = 1,
  PENTAD_STATUS_ALLOC_FAILED              = 2,
  PENTAD_STATUS_INVALID_VALUE             = 3,
  PENTAD_STATUS_ARCH_MISMATCH             = 4,
  PENTAD_STATUS_MAPPING_ERROR             = 5,
  PENTAD_STATUS_EXECUTION_FAILED          = 6,
  PENTAD_STATUS_INTERNAL_ERROR            = 7,
  PENTAD_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
  PENTAD_STATUS_ZERO_PIVOT                = 9
} pentadStatus_t;

#endif
