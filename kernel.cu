/*
 * Copyright (C) 2024 Forefront Industries, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_fp16.h"
#include <cuda_pipeline.h>
#include <stdio.h>
#include <cmath>
#include <tuple>
#include "configs.cu"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#define load(x) __ldcg(x)
#define store(x, value) __stcs(x, value)
#define div_ru(a, b) (a + b - 1) / b
#define div_rd(a, b) a / b
#define VOLATILE
#define DEBUG

#define CUDA_DEVICE_INLINE __device__ __forceinline__
#define ENABLE_L2_PREFETCH true

#define LAYOUT_C true
#define LAYOUT_F false
#define TRANSFORM_N true
#define TRANSFORM_T false
#define QUANTIZED true
#define NOT_QUANTIZED false
#define ACC_MODE_HALF 16
#define ACC_MODE_FLOAT 32
typedef bool MATRIX_LAYOUT;
typedef bool MATRIX_TRANSFORM;
typedef bool MATRIX_QUANTIZE_STATUS;

using typeA = uint8_t;
using typeB = half;
using typeCode = half;
using typeAbsMax = half;
using typeAcc = float;

constexpr int accMode = 32;
constexpr int accBytes = 4; // accumulate in fp32

typedef struct __builtin_align__(8)
{
    half x1, x2, x3, x4;
}
half4;

typedef struct __builtin_align__(16)
{
    half x1, x2, x3, x4, x5, x6, x7, x8;
}
half8;

typedef struct __builtin_align__(16)
{
    unsigned char x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16;
}
uchar16;

typedef struct __builtin_align__(16)
{
    char x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16;
}
char16;

typedef struct
{
    int x;
} Coord1D;

typedef struct
{
    int x, y;
} Coord2D;

typedef struct
{
    int x, y, z;
} Coord3D;

typedef struct
{
    int x, y, z, t;
} Coord4D;

typedef struct
{
    int x, y, z, t, u;
} Coord5D;

typedef union
{
    signed char as_int8[1];
    unsigned char as_uint8[1];
} Data1B;

typedef union
{
    short as_int16[1];
    unsigned short as_uint16[1];
    signed char as_int8[2];
    unsigned char as_uint8[2];
    half as_half[1];
} Data2B;

typedef union
{
    int as_int32[1];
    unsigned int as_uint32[1];
    short as_int16[2];
    unsigned short as_uint16[2];
    signed char as_int8[4];
    unsigned char as_uint8[4];
    float as_float[1];
    half2 as_half2[1];
    half as_half[2];
} Data4B;

typedef union
{
    long long as_int64[1];
    unsigned long long as_uint64[1];
    int as_int32[2];
    unsigned int as_uint32[2];
    short as_int16[4];
    unsigned short as_uint16[4];
    signed char as_int8[8];
    unsigned char as_uint8[8];
    double as_double[1];
    half4 as_half4[1];
    float2 as_float2[1];
    float as_float[2];
    half2 as_half2[2];
    half as_half[4];
} Data8B;

typedef union
{
    uchar16 as_uchar16[1];
    char16 as_char16[1];
    long long as_int64[2];
    unsigned long long as_uint64[2];
    int as_int32[4];
    unsigned int as_uint32[4];
    short as_int16[8];
    unsigned short as_uint16[8];
    signed char as_int8[16];
    unsigned char as_uint8[16];
    half8 as_half8[1];
    double as_double[2];
    half4 as_half4[2];
    float2 as_float2[2];
    float as_float[4];
    half2 as_half2[4];
    half as_half[8];
} Data16B;

#if (!defined(__clang__) && __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)
extern "C"
{
    __device__ uint32_t __nvvm_get_smem_pointer(void *);
}
#endif

CUDA_DEVICE_INLINE
unsigned get_smem_pointer(void const *ptr)
{
#if (!defined(__clang__) && defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11)
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
#elif (!defined(__clang__) && defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2)
    return __nvvm_get_smem_pointer(ptr);
#elif defined(__CUDA_ARCH__)
    uint32_t smem_ptr;
    asm(
        "{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n"
        : "=r"(smem_ptr) : "l"(ptr));
    return smem_ptr;
#else
    return 0;
#endif
}

CUDA_DEVICE_INLINE
void ldsm_x1_C(
    void *srcPtr,
    Data4B dest[1])
{
    unsigned int srcAddr = get_smem_pointer(srcPtr);
    asm volatile(
        "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];"
        : "=r"(dest[0].as_int32[0])
        : "r"(srcAddr));
}

CUDA_DEVICE_INLINE
void ldsm_x2_C(
    void *srcPtr,
    Data4B dest[2])
{
    unsigned int srcAddr = get_smem_pointer(srcPtr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];"
        : "=r"(dest[0].as_int32[0]), "=r"(dest[1].as_int32[0])
        : "r"(srcAddr));
}

CUDA_DEVICE_INLINE
void ldsm_x4_C(
    void *srcPtr,
    Data4B dest[4])
{
    unsigned int srcAddr = get_smem_pointer(srcPtr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(dest[0].as_int32[0]), "=r"(dest[1].as_int32[0]), "=r"(dest[2].as_int32[0]), "=r"(dest[3].as_int32[0])
        : "r"(srcAddr));
}

CUDA_DEVICE_INLINE
void ldsm_x1_F(
    void *srcPtr,
    Data4B dest[1])
{
    unsigned int srcAddr = get_smem_pointer(srcPtr);
    asm volatile(
        "ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];"
        : "=r"(dest[0].as_int32[0])
        : "r"(srcAddr));
}

CUDA_DEVICE_INLINE
void ldsm_x2_F(
    void *srcPtr,
    Data4B dest[2])
{
    unsigned int srcAddr = get_smem_pointer(srcPtr);
    asm volatile(
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];"
        : "=r"(dest[0].as_int32[0]), "=r"(dest[1].as_int32[0])
        : "r"(srcAddr));
}

CUDA_DEVICE_INLINE
void ldsm_x4_F(
    void *srcPtr,
    Data4B dest[4])
{
    unsigned int srcAddr = get_smem_pointer(srcPtr);
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(dest[0].as_int32[0]), "=r"(dest[1].as_int32[0]), "=r"(dest[2].as_int32[0]), "=r"(dest[3].as_int32[0])
        : "r"(srcAddr));
}

template <int matrixCount, MATRIX_LAYOUT layout>
CUDA_DEVICE_INLINE void ldsm(
    void *srcPtr,
    Data4B dest[matrixCount])
{
    if (matrixCount == 1 && layout == LAYOUT_C)
        ldsm_x1_C(srcPtr, dest);

    else if (matrixCount == 2 && layout == LAYOUT_C)
        ldsm_x2_C(srcPtr, dest);

    else if (matrixCount == 4 && layout == LAYOUT_C)
        ldsm_x4_C(srcPtr, dest);

    else if (matrixCount == 1 && layout == LAYOUT_F)
        ldsm_x1_F(srcPtr, dest);

    else if (matrixCount == 2 && layout == LAYOUT_F)
        ldsm_x2_F(srcPtr, dest);

    else if (matrixCount == 4 && layout == LAYOUT_F)
        ldsm_x4_F(srcPtr, dest);
}

template <int size>
CUDA_DEVICE_INLINE void ldgsts_ca(
    void const *srcPtr,
    void *destPtr,
    bool p = true)
{
    unsigned smemPtr = get_smem_pointer(destPtr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)p),
        "r"(smemPtr), "l"(srcPtr), "n"(size));
}

template <int size>
CUDA_DEVICE_INLINE void ldgsts_ca_prefetch(
    void const *srcPtr,
    void *destPtr,
    bool p = true)
{
    unsigned smemPtr = get_smem_pointer(destPtr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;\n"
        "}\n" ::"r"((int)p),
        "r"(smemPtr), "l"(srcPtr), "n"(size));
}

template <int size>
CUDA_DEVICE_INLINE void ldgsts_cg(
    void const *srcPtr,
    void *destPtr,
    bool p = true)
{
    unsigned smemPtr = get_smem_pointer(destPtr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" ::"r"((int)p),
        "r"(smemPtr), "l"(srcPtr), "n"(size));
}

template <int size>
CUDA_DEVICE_INLINE void ldgsts_cg_prefetch(
    void const *srcPtr,
    void *destPtr,
    bool p = true)
{
    unsigned smemPtr = get_smem_pointer(destPtr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
        "  @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
        "}\n" ::"r"((int)p),
        "r"(smemPtr), "l"(srcPtr), "n"(size));
}

template <int size, bool cacheAllways, bool prefetchL2>
CUDA_DEVICE_INLINE void ldgsts(
    // void const* srcPtr,
    void const *srcPtr,
    void *destPtr,
    bool p = true)
{
    if (cacheAllways)
        if (prefetchL2)
            ldgsts_ca_prefetch<size>(srcPtr, destPtr, p);
        else
            ldgsts_ca<size>(srcPtr, destPtr, p);
    else if (prefetchL2)
        ldgsts_cg_prefetch<size>(srcPtr, destPtr, p);
    else
        ldgsts_cg<size>(srcPtr, destPtr, p);
}
CUDA_DEVICE_INLINE
void mma_m16n8k16_fp16_CF(
    Data4B fragA[4],
    Data4B fragB[2],
    Data4B accumulator[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};"
        : "+r"(accumulator[0].as_int32[0]), "+r"(accumulator[1].as_int32[0])
        : "r"(fragA[0].as_int32[0]), "r"(fragA[1].as_int32[0]), "r"(fragA[2].as_int32[0]), "r"(fragA[3].as_int32[0]), "r"(fragB[0].as_int32[0]), "r"(fragB[1].as_int32[0]));
}

CUDA_DEVICE_INLINE
void mma_m16n8k8_fp16_CF(
    Data4B fragA[2],
    Data4B fragB[1],
    Data4B accumulator[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%0, %1};"
        : "+r"(accumulator[0].as_int32[0]), "+r"(accumulator[1].as_int32[0])
        : "r"(fragA[0].as_int32[0]), "r"(fragA[1].as_int32[0]), "r"(fragB[0].as_int32[0]));
}

CUDA_DEVICE_INLINE
void mma_m16n8k8_fp32_CF(
    Data4B fragA[2],
    Data4B fragB[1],
    Data4B accumulator[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+r"(accumulator[0].as_int32[0]), "+r"(accumulator[1].as_int32[0]), "+r"(accumulator[2].as_int32[0]), "+r"(accumulator[3].as_int32[0])
        : "r"(fragA[0].as_int32[0]), "r"(fragA[1].as_int32[0]), "r"(fragB[0].as_int32[0]));
}

CUDA_DEVICE_INLINE
void mma_m16n8k16_fp32_CF(
    Data4B fragA[4],
    Data4B fragB[2],
    Data4B accumulator[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+r"(accumulator[0].as_int32[0]), "+r"(accumulator[1].as_int32[0]), "+r"(accumulator[2].as_int32[0]), "+r"(accumulator[3].as_int32[0])
        : "r"(fragA[0].as_int32[0]), "r"(fragA[1].as_int32[0]), "r"(fragA[2].as_int32[0]), "r"(fragA[3].as_int32[0]), "r"(fragB[0].as_int32[0]), "r"(fragB[1].as_int32[0]));
}

#define mma_param_is_equal(left, p1, p2, p3, p4, p5) \
    (left.mmaSizeM == p1 &&                          \
     left.mmaSizeN == p2 &&                          \
     left.mmaSizeK == p3 &&                          \
     left.layoutA == p4 &&                           \
     left.layoutB == p5)

typedef struct
{
    int mmaSizeM, mmaSizeN, mmaSizeK;
    MATRIX_LAYOUT layoutA, layoutB;
} MmaParam;

template <
    int mmaSizeM,
    int mmaSizeN,
    int mmaSizeK,
    MATRIX_LAYOUT layoutA,
    MATRIX_LAYOUT layoutB,
    int outPrecision>
CUDA_DEVICE_INLINE void mma(
    Data4B fragA[(mmaSizeM / 8) * (mmaSizeK / 8)],
    Data4B fragB[(mmaSizeN / 8) * (mmaSizeK / 8)],
    Data4B accumulator[(mmaSizeM / 8) * (mmaSizeN / 8) * (accMode == 32 ? 2 : 1)])
{
    constexpr MmaParam param = {mmaSizeM, mmaSizeN, mmaSizeK, layoutA, layoutB};
    if constexpr (accMode == 32)
    {
        if (mma_param_is_equal(param, 16, 8, 8, LAYOUT_C, LAYOUT_F))
            mma_m16n8k8_fp32_CF(fragA, fragB, accumulator);

        else if (mma_param_is_equal(param, 16, 8, 16, LAYOUT_C, LAYOUT_F))
            mma_m16n8k16_fp32_CF(fragA, fragB, accumulator);
    }
    else if constexpr (accMode == 16)
    {
        if (mma_param_is_equal(param, 16, 8, 8, LAYOUT_C, LAYOUT_F))
            mma_m16n8k8_fp16_CF(fragA, fragB, accumulator);

        else if (mma_param_is_equal(param, 16, 8, 16, LAYOUT_C, LAYOUT_F))
            mma_m16n8k16_fp16_CF(fragA, fragB, accumulator);
    }
}

template <
    int mmaSizeM,
    int mmaSizeN,
    int mmaSizeK,
    MATRIX_LAYOUT layoutA,
    MATRIX_LAYOUT layoutB,
    int outPrecision>
CUDA_DEVICE_INLINE void ntcmma(
    Data4B fragA[(mmaSizeM / 8) * (mmaSizeK / 8)],
    Data4B fragB[(mmaSizeN / 8) * (mmaSizeK / 8)],
    Data4B accumulator[(mmaSizeM / 8) * (mmaSizeN / 8) * (outPrecision == 32 ? 2 : 1)])
{
    int tid = threadIdx.x;
    int lid = tid % 32;
    constexpr Coord3D mmaMatrixCount = {mmaSizeM / 8, mmaSizeN / 8, mmaSizeK / 8};
    Coord2D laneIdx4 = {lid / 4, lid % 4};

#pragma unroll
    for (int om = 0; om < mmaMatrixCount.x; om++)
    {
#pragma unroll
        for (int on = 0; on < mmaMatrixCount.y; on++)
        {
#pragma unroll
            for (int ok = 0; ok < mmaMatrixCount.z; ok++)
            {
                half2 currentFragA = fragA[om * mmaMatrixCount.z + ok].as_half2[0];
                half2 currentFragB = fragB[on * mmaMatrixCount.z + ok].as_half2[0];
#pragma unroll
                for (int im = 0; im < 8; im++)
                {
#pragma unroll
                    for (int in = 0; in < 8; in++)
                    {
#pragma unroll
                        for (int ik = 0; ik < 4; ik++)
                        {
                            half2 swappedFragA = __shfl_sync(0xffffffff, currentFragA, im * 4 + ik);
                            half2 swappedFragB = __shfl_sync(0xffffffff, currentFragB, in * 4 + ik);
                            accumulator[om * mmaMatrixCount.y + on].as_half2[0] = __hfma2(swappedFragA, swappedFragB, accumulator[om * mmaMatrixCount.y + on].as_half2[0]);
                        }
                    }
                }
            }
        }
    }
}
template <
    typename T>
class SmemTensor0D
{
public:
    VOLATILE T *endPtr;
    VOLATILE T *startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor0D(VOLATILE void *smemPtr)
        : startPtr(reinterpret_cast<T *>(smemPtr)), endPtr(reinterpret_cast<T *>(smemPtr))
    {
    }

    CUDA_DEVICE_INLINE
    T get()
    {
        return startPtr[0];
    }

    CUDA_DEVICE_INLINE
    T *get_ptr()
    {
        return startPtr;
    }

    CUDA_DEVICE_INLINE
    void set(const T value)
    {
        startPtr[0] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
        U
        get_reinterpreted()
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        return newPtr[0];
    }

    template <typename U>
    CUDA_DEVICE_INLINE void set_reinterpreted(U value)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        newPtr[0] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
        U *
        get_ptr_reinterpreted()
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        return &newPtr[0];
    }
};

template <
    typename T,
    int ShapeX>
class SmemTensor1D
{
public:
    VOLATILE T *endPtr;
    VOLATILE T *startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor1D(VOLATILE void *smemPtr)
        : startPtr(reinterpret_cast<T *>(smemPtr)), endPtr(reinterpret_cast<T *>(smemPtr) + shape().x)
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x)
    {
        return startPtr[x];
    }

    CUDA_DEVICE_INLINE
    T *get_ptr(int x)
    {
        return &startPtr[x];
    }

    CUDA_DEVICE_INLINE
    void set(int x, const T value)
    {
        startPtr[x] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
        U
        get_reinterpreted(int x)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        return newPtr[x];
    }

    template <typename U>
    CUDA_DEVICE_INLINE void set_reinterpreted(int x, U value)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        newPtr[x] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
        U *
        get_ptr_reinterpreted(int x)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        return &newPtr[x];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord1D shape()
    {
        return {ShapeX};
    }
};

template <
    typename T,
    int ShapeX,
    int ShapeY>
class SmemTensor2D
{
public:
    VOLATILE T *endPtr;
    VOLATILE T *startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor2D(VOLATILE void *smemPtr)
        : startPtr(reinterpret_cast<T *>(smemPtr)), endPtr(reinterpret_cast<T *>(smemPtr) + shape().x * shape().y)
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x, int y)
    {
        return startPtr[x * stride().x + y];
    }

    CUDA_DEVICE_INLINE
    T *get_ptr(int x, int y)
    {
        return &startPtr[x * stride().x + y];
    }

    CUDA_DEVICE_INLINE
    void set(int x, int y, const T value)
    {
        startPtr[x * stride().x + y] = value;
    }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeY> get_child(int x)
    {
        SmemTensor1D<T, ShapeY> child(
            &startPtr[x * stride().x]);
        return child;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
        U
        get_reinterpreted(int x, int y)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        return newPtr[(x * stride().x) * sizeof(T) / sizeof(U) +
                      y];
    }

    template <typename U>
    CUDA_DEVICE_INLINE void set_reinterpreted(int x, int y, U value)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        newPtr[(x * stride().x) * sizeof(T) / sizeof(U) +
               y] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
        U *
        get_ptr_reinterpreted(int x, int y)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        return &newPtr[(x * stride().x) * sizeof(T) / sizeof(U) +
                       y];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D shape()
    {
        return {ShapeX, ShapeY};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord1D stride()
    {
        return {ShapeY};
    }
};

template <
    typename T,
    int ShapeX,
    int ShapeY,
    int ShapeZ>
class SmemTensor3D
{
public:
    VOLATILE T *endPtr;
    VOLATILE T *startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor3D(VOLATILE void *smemPtr)
        : startPtr(reinterpret_cast<T *>(smemPtr)), endPtr(reinterpret_cast<T *>(smemPtr) + shape().x * shape().y * shape().z)
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x, int y, int z)
    {
        return startPtr[x * stride().x + y * stride().y + z];
    }

    CUDA_DEVICE_INLINE
    T *get_ptr(int x, int y, int z)
    {
        return &startPtr[x * stride().x + y * stride().y + z];
    }

    CUDA_DEVICE_INLINE
    void set(int x, int y, int z, const T value)
    {
        startPtr[x * stride().x + y * stride().y + z] = value;
    }

    CUDA_DEVICE_INLINE
    SmemTensor2D<T, ShapeY, ShapeZ> get_child(int x)
    {
        SmemTensor2D<T, ShapeY, ShapeZ> child(
            &startPtr[x * stride().x]);
        return child;
    }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeZ> get_child(int x, int y)
    {
        SmemTensor1D<T, ShapeZ> child(
            &startPtr[x * stride().x + y * stride().y]);
        return child;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
        U
        get_reinterpreted(int x, int y, int z)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        return newPtr[(x * stride().x +
                       y * stride().y) *
                          sizeof(T) / sizeof(U) +
                      z];
    }

    template <typename U>
    CUDA_DEVICE_INLINE void set_reinterpreted(int x, int y, int z, U value)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        newPtr[(x * stride().x +
                y * stride().y) *
                   sizeof(T) / sizeof(U) +
               z] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
        U *
        get_ptr_reinterpreted(int x, int y, int z)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        return &newPtr[(x * stride().x +
                        y * stride().y) *
                           sizeof(T) / sizeof(U) +
                       z];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord3D shape()
    {
        return {ShapeX, ShapeY, ShapeZ};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D stride()
    {
        return {ShapeY * ShapeZ, ShapeZ};
    }
};

template <
    typename T,
    int ShapeX,
    int ShapeY,
    int ShapeZ,
    int ShapeT>
class SmemTensor4D
{
public:
    VOLATILE T *endPtr;
    VOLATILE T *startPtr;

    CUDA_DEVICE_INLINE
    SmemTensor4D(VOLATILE void *smemPtr)
        : startPtr(reinterpret_cast<T *>(smemPtr)), endPtr(&reinterpret_cast<T *>(smemPtr)[shape().x * shape().y * shape().z * shape().t])
    {
    }

    CUDA_DEVICE_INLINE
    T get(int x, int y, int z, int t)
    {
        return startPtr[x * stride().x +
                        y * stride().y +
                        z * stride().z +
                        t];
    }

    CUDA_DEVICE_INLINE
    T *get_ptr(int x, int y, int z, int t)
    {
        return &startPtr[x * stride().x +
                         y * stride().y +
                         z * stride().z +
                         t];
    }

    CUDA_DEVICE_INLINE
    void set(int x, int y, int z, int t, const T value)
    {
        startPtr[x * stride().x +
                 y * stride().y +
                 z * stride().z +
                 t] = value;
    }

    CUDA_DEVICE_INLINE
    SmemTensor3D<T, ShapeY, ShapeZ, ShapeT> get_child(int x)
    {
        SmemTensor3D<T, ShapeY, ShapeZ, ShapeT> child(
            &startPtr[x * stride().x]);
        return child;
    }

    CUDA_DEVICE_INLINE
    SmemTensor2D<T, ShapeZ, ShapeT> get_child(int x, int y)
    {
        SmemTensor2D<T, ShapeZ, ShapeT> child(
            &startPtr[x * stride().x + y * stride().y]);
        return child;
    }

    CUDA_DEVICE_INLINE
    SmemTensor1D<T, ShapeT> get_child(int x, int y, int z)
    {
        SmemTensor1D<T, ShapeT> child(
            &startPtr[x * stride().x + y * stride().y + z * stride().z]);
        return child;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
        U
        get_reinterpreted(int x, int y, int z, int t)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        return newPtr[(x * stride().x +
                       y * stride().y +
                       z * stride().z) *
                          sizeof(T) / sizeof(U) +
                      t];
    }

    template <typename U>
    CUDA_DEVICE_INLINE void set_reinterpreted(int x, int y, int z, int t, U value)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        newPtr[(x * stride().x +
                y * stride().y +
                z * stride().z) *
                   sizeof(T) / sizeof(U) +
               t] = value;
    }

    template <typename U>
    CUDA_DEVICE_INLINE
        U *
        get_ptr_reinterpreted(int x, int y, int z, int t)
    {
        U *newPtr = reinterpret_cast<U *>(startPtr);
        return &newPtr[(x * stride().x +
                        y * stride().y +
                        z * stride().z) *
                           sizeof(T) / sizeof(U) +
                       t];
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord4D shape()
    {
        return {ShapeX, ShapeY, ShapeZ, ShapeT};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord3D stride()
    {
        return {
            ShapeY * ShapeZ * ShapeT,
            ShapeZ * ShapeT,
            ShapeT};
    }
};

template <
    typename T,
    int TileN,     // non contiguous dimention
    int TileC,     // contiguous dimention
    int AccessSize // number of type T elements to read per access. Should be 8 fp16 or 16 int8 for mma.sync
                   // MATRIX_LAYOUT Layout
    >
class TileIndexConverter
{
public:
    CUDA_DEVICE_INLINE
    TileIndexConverter()
    {
    }

    CUDA_DEVICE_INLINE
    Coord2D convert(const Coord2D logicalIdx)
    {
        if (splits() == 1)
        {
            return {
                logicalIdx.x,
                (logicalIdx.y + (logicalIdx.x % indent_count()) * AccessSize) % physical_shape().y};
        }
        else
        {
            Coord2D tempIdx = {
                logicalIdx.x % physical_shape().x,
                (logicalIdx.y + (logicalIdx.x / physical_shape().x) * logical_shape().y)};
            return {
                tempIdx.x,
                (tempIdx.y + (tempIdx.x % indent_count()) * AccessSize) % physical_shape().y};
        }
    }

    CUDA_DEVICE_INLINE
    Coord2D convert_reinterpreted(const Coord2D logicalIdx, const int newSize)
    {
        Coord2D result = convert({logicalIdx.x, logicalIdx.y * newSize / sizeof(T)});
        result.y = result.y * sizeof(T) / newSize;
        return result;
    }

    CUDA_DEVICE_INLINE
    int convert_to_offset(const Coord2D logicalIdx)
    {
        Coord2D physicalIdx = convert(logicalIdx);
        return physicalIdx.x * physical_shape().y + physicalIdx.y;
    }

    CUDA_DEVICE_INLINE
    int convert_to_offset_reinterpreted(const Coord2D logicalIdx, const int newSize)
    {
        Coord2D physicalIdx = convert_reinterpreted(logicalIdx, newSize);
        return physicalIdx.x * (physical_shape_bytes().y / newSize) + physicalIdx.y;
    }

    CUDA_DEVICE_INLINE
    static constexpr int splits()
    {
        return logical_shape_bytes().y < 128 ? 128 / logical_shape_bytes().y : 1;
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D physical_shape()
    {
        // when tileC * sizeof(T) < 128 Bytes, in order to make the access bank conflict free, we stplit TileN (128 / (tileC * sizeof(T))) times
        // for example, when tileC * sizeof(T) == 32 bytes, TileN will be splitted into 4.
        Coord2D result = {TileN / splits(), TileC * splits()}; // 64, 128
        return result;
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D logical_shape()
    {
        return {TileN, TileC}; // 256, 32
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D physical_shape_bytes()
    {
        return {physical_shape().x, physical_shape().y * sizeof(T)};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D logical_shape_bytes()
    {
        return {logical_shape().x, logical_shape().y * sizeof(T)};
    }

    CUDA_DEVICE_INLINE
    static constexpr int access_size_bytes()
    {
        return AccessSize * sizeof(T);
    }

    CUDA_DEVICE_INLINE
    static constexpr int indent_count()
    {
        return 128 / access_size_bytes(); // 8
    }
};

template <
    int BlockTileM,
    int BlockTileN,
    int BlockTileK,
    MATRIX_TRANSFORM TransformA,
    MATRIX_TRANSFORM TransformB,
    MATRIX_QUANTIZE_STATUS QuantStatA,
    MATRIX_QUANTIZE_STATUS QuantStatB,
    typename TypeA,
    typename TypeB,
    int AlignSizeA,
    int AlignSizeB,
    int ContiguousBytesA,
    int ContiguousBytesB,
    int WarpCountM,
    int WarpCountN,
    int MmaSizeM,
    int MmaSizeN,
    int MmaSizeK,
    int WarpMmaCountM,
    int WarpMmaCountN,
    int WarpMmaCountK,
    int Stages,
    int AbsMaxPerBlock,
    int CodeSize,
    int PhysicalShapeAX,
    int PhysicalShapeAY,
    int PhysicalShapeBX,
    int PhysicalShapeBY,
    bool IsSafe,
    int PipelineStrat,
    int AccMode>
class Deq4GemmIterator
{
public:
    CUDA_DEVICE_INLINE
    Deq4GemmIterator(
        const void *ptrA,
        const void *ptrB,
        Coord3D problemSize,
        Coord3D globalStart, Coord3D globalEnd, Coord3D leadingDims)
        : _ptrA(ptrA), _ptrB(ptrB), _nMainIter(((globalEnd.z - globalStart.z) + BlockTileK - 1) / BlockTileK), _tid(threadIdx.x), _wid(__shfl_sync(0xffffffff, threadIdx.x / 32, 0)),
          _lid(threadIdx.x % 32), _problemSize(problemSize), _globalStart(globalStart), _globalEnd(globalEnd), _leadingDims(leadingDims)

    {

        _strideA = {leadingDims.x};
        _strideB = {leadingDims.y};

        _warpIdx = {_wid / WarpCountN, _wid % WarpCountN};
        _laneIdx8 = {_lid / 8, _lid % 8};
        _laneIdx4 = {_lid / 4, _lid % 4};
        _laneIdx8r02 = {_laneIdx8.x / 2, _laneIdx8.x % 2};
        _warpLogicalIdxStart = {_warpIdx.x * WarpMmaCountM * (MmaSizeM / 8), _warpIdx.y * WarpMmaCountN * (MmaSizeN / 8)};
        _laneIdxA = {_lid / (ContiguousBytesA / align_size_bytes_A()), _lid % (ContiguousBytesA / align_size_bytes_A())};
        _laneIdxB = {_lid / (ContiguousBytesB / align_size_bytes_B()), _lid % (ContiguousBytesB / align_size_bytes_B())};

        _validTileSizeA = logical_tile_size_A();
        if (TransformA == TRANSFORM_N)
        {
            if (global_start_A().x + logical_tile_size_A().x > global_end_A().x)
            {
                _validTileSizeA.x = (global_end_A().x - global_start_A().x);
            }
        }
        else
        {
            if (global_start_A().y + logical_tile_size_A().y > global_end_A().y)
            {
                _validTileSizeA.y = (global_end_A().y - global_start_A().y);
            }
        }

        _validTileSizeB = logical_tile_size_B();
        if (TransformB == TRANSFORM_N)
        {
            if (global_start_B().y + logical_tile_size_B().y > global_end_B().y)
            {
                _validTileSizeB.y = (global_end_B().y - global_start_B().y);
            }
        }
        else
        {
            if (global_start_B().x + logical_tile_size_B().x > global_end_B().x)
            {
                _validTileSizeB.x = (global_end_B().x - global_start_B().x);
            }
        }
    }

    CUDA_DEVICE_INLINE
    void run_nstage(
        // SmemTensor3D<TypeA, Stages, PhysicalShapeAX, PhysicalShapeAY> smemA,
        SmemTensor4D<TypeA, Stages, BlockTileM / 8, BlockTileK / 8, 32> smemA,
        SmemTensor3D<TypeB, Stages, PhysicalShapeBX, PhysicalShapeBY> smemB,
        SmemTensor1D<half, CodeSize> smemCode,
        SmemTensor1D<half, AbsMaxPerBlock> smemAbsMax,
        Data4B accumulator[WarpMmaCountM][WarpMmaCountN][accBytes],
        int &useless)
    {
        // const int tid = threadIdx.x;
        // const int lid = tid % 32;
        // const int wid = __shfl_sync(0xffffffff, tid / 32, 0);
        constexpr Coord3D mmaSize = {MmaSizeM, MmaSizeN, MmaSizeK};
        constexpr Coord3D mmaMatrixCount = {mmaSize.x / 8, mmaSize.y / 8, mmaSize.z / 8};
        constexpr Coord3D warpMmaCount = {WarpMmaCountM, WarpMmaCountN, WarpMmaCountK}; // [4, 8, 2]
        constexpr Coord2D warpCount = {WarpCountM, WarpCountN};

        Data1B bufferA[warpMmaCount.z][warpMmaCount.x][4];
        Data4B bufferB[warpMmaCount.z][warpMmaCount.y][mmaMatrixCount.y * mmaMatrixCount.z / (QuantStatB == QUANTIZED ? 2 : 1)];
        // Data4B bufferB[][];

        Data4B fragA[warpMmaCount.z][warpMmaCount.x][mmaMatrixCount.x * mmaMatrixCount.z];
        Data4B fragB[warpMmaCount.z][warpMmaCount.y][mmaMatrixCount.y * mmaMatrixCount.z];

        const Coord2D logicalBaseIdxA = logical_base_idx_A();
        const Coord2D logicalBaseIdxB = logical_base_idx_B();

#pragma unroll
        for (int s = 0; s < Stages - 1; s++)
        {
            // TODO: LDGSTS LOGIC HERE
            deq4_ldgsts_A(smemA.get_child(s));
            deq4_ldgsts_B(smemB.get_child(s));
            deq4_ldgsts_move_to_next_tile();
            __pipeline_commit();
        }
        __pipeline_wait_prior(Stages - 2);
        __syncthreads();

        auto currentStageSmemPtrA = smemA.get_child(0);
        TypeB *currentStageSmemPtrB = smemB.get_child(0).startPtr;

#pragma unroll
        for (int i = 0; i < warpMmaCount.x; i++)
        {
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
        }
#pragma unroll
        for (int i = 0; i < warpMmaCount.y / 2; i++)
        {
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
        }
#pragma unroll
        for (int mi = 0; mi < warpMmaCount.x; mi++)
        {
            deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 0, mi);
        }
#pragma unroll
        for (int ni = 0; ni < warpMmaCount.y; ni++)
        {
            deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 0, ni);
        }

        for (int ki = 0; ki < _nMainIter; ki++)
        {
            int currentStage = ki % Stages;
#pragma unroll
            for (int i = 0; i < warpMmaCount.x; i++)
            {
                deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 1, i);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.y / 2; i++)
            {
                deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 1, i);
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < warpMmaCount.y; ni++)
                {
                    // HMMA(mi, ni, 0);
                    deq4_hmma(fragA, fragB, accumulator, mi, ni, 0);
                }
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
                deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 1, mi);
            }
#pragma unroll
            for (int ni = 0; ni < warpMmaCount.y; ni++)
            {
                deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 1, ni);
            }

            if (ki < _nMainIter - (Stages - 1))
            {
                // TODO: LDGSTS logic here
                deq4_ldgsts_A(smemA.get_child((currentStage + Stages - 1) % Stages));
                deq4_ldgsts_B(smemB.get_child((currentStage + Stages - 1) % Stages));
                deq4_ldgsts_move_to_next_tile();
                __pipeline_commit();
            }
            __pipeline_wait_prior(Stages - 2);
            __syncthreads();
            currentStageSmemPtrA = smemA.get_child((currentStage + 1) % Stages);
            currentStageSmemPtrB = smemB.get_child((currentStage + 1) % Stages).startPtr;

#pragma unroll
            for (int i = 0; i < warpMmaCount.x; i++)
            {
                deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.y / 2; i++)
            {
                deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < warpMmaCount.y; ni++)
                {
                    deq4_hmma(fragA, fragB, accumulator, mi, ni, 1);
                }
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
                deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 0, mi);
            }

#pragma unroll
            for (int ni = 0; ni < warpMmaCount.y; ni++)
            {
                deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 0, ni);
            }
        }
    }

    CUDA_DEVICE_INLINE
    void run_nstage_v2(
        // SmemTensor3D<TypeA, Stages, PhysicalShapeAX, PhysicalShapeAY> smemA,
        SmemTensor4D<TypeA, Stages, BlockTileM / 8, BlockTileK / 8, 32> smemA,
        SmemTensor3D<TypeB, Stages, PhysicalShapeBX, PhysicalShapeBY> smemB,
        SmemTensor1D<half, CodeSize> smemCode,
        SmemTensor1D<half, AbsMaxPerBlock> smemAbsMax,
        Data4B accumulator[WarpMmaCountM][WarpMmaCountN][accBytes],
        int &useless)
    {
        // const int tid = threadIdx.x;
        // const int lid = tid % 32;
        // const int wid = __shfl_sync(0xffffffff, tid / 32, 0);
        constexpr Coord3D mmaSize = {MmaSizeM, MmaSizeN, MmaSizeK};
        constexpr Coord3D mmaMatrixCount = {mmaSize.x / 8, mmaSize.y / 8, mmaSize.z / 8};
        constexpr Coord3D warpMmaCount = {WarpMmaCountM, WarpMmaCountN, WarpMmaCountK}; // [4, 8, 2]
        constexpr Coord2D warpCount = {WarpCountM, WarpCountN};
        Coord2D warpIdx = {_wid / warpCount.y, _wid % warpCount.y};
        Coord2D laneIdx = {_lid / 8, _lid % 8};

        Data1B bufferA[warpMmaCount.z][warpMmaCount.x][4];
        Data4B bufferB[warpMmaCount.z][warpMmaCount.y][mmaMatrixCount.y * mmaMatrixCount.z / (QuantStatB == QUANTIZED ? 2 : 1)];
        // Data4B bufferB[][];

        Data4B fragA[warpMmaCount.z][warpMmaCount.x][mmaMatrixCount.x * mmaMatrixCount.z];
        Data4B fragB[warpMmaCount.z][warpMmaCount.y][mmaMatrixCount.y * mmaMatrixCount.z];

        const Coord2D logicalBaseIdxA = logical_base_idx_A();

        const Coord2D logicalBaseIdxB = logical_base_idx_B();
#pragma unroll
        for (int s = 0; s < Stages - 1; s++)
        {
            // TODO: LDGSTS LOGIC HERE
            deq4_ldgsts_A(smemA.get_child(s));
            deq4_ldgsts_B(smemB.get_child(s));
            deq4_ldgsts_move_to_next_tile();
            __pipeline_commit();
        }
        __pipeline_wait_prior(Stages - 2);
        __syncthreads();

        auto currentStageSmemPtrA = smemA.get_child(0);
        TypeB *currentStageSmemPtrB = smemB.get_child(0).startPtr;

#pragma unroll
        for (int i = 0; i < warpMmaCount.x; i++)
        {
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 1, i);
        }
#pragma unroll
        for (int i = 0; i < warpMmaCount.y / 2; i++)
        {
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 1, i);
        }

#pragma unroll
        for (int mi = 0; mi < warpMmaCount.x; mi++)
        {
            deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 0, mi);
        }
#pragma unroll
        for (int ni = 0; ni < warpMmaCount.y; ni++)
        {
            deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 0, ni);
        }

        for (int ki = 0; ki < _nMainIter; ki++)
        {
            int currentStage = ki % Stages;
#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < warpMmaCount.y; ni++)
                {
                    // HMMA(mi, ni, 0);
                    deq4_hmma(fragA, fragB, accumulator, mi, ni, 0);
                }
            }

            __pipeline_wait_prior(Stages - 2);
            __syncthreads();

            currentStageSmemPtrA = smemA.get_child((currentStage + 1) % Stages);
            currentStageSmemPtrB = smemB.get_child((currentStage + 1) % Stages).startPtr;

#pragma unroll
            for (int i = 0; i < warpMmaCount.x; i++)
            {
                deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.y / 2; i++)
            {
                deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
                deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 1, mi);
            }
#pragma unroll
            for (int ni = 0; ni < warpMmaCount.y; ni++)
            {
                deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 1, ni);
            }

            if (ki < _nMainIter - (Stages - 1))
            {
                // TODO: LDGSTS logic here
                deq4_ldgsts_A(smemA.get_child((currentStage + Stages - 1) % Stages));
                deq4_ldgsts_B(smemB.get_child((currentStage + Stages - 1) % Stages));
                deq4_ldgsts_move_to_next_tile();
                __pipeline_commit();
            }
            useless = clock();

#pragma unroll
            for (int i = 0; i < warpMmaCount.x; i++)
            {
                deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 1, i);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.y / 2; i++)
            {
                deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 1, i);
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < warpMmaCount.y; ni++)
                {
                    deq4_hmma(fragA, fragB, accumulator, mi, ni, 1);
                }
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
                deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 0, mi);
            }

#pragma unroll
            for (int ni = 0; ni < warpMmaCount.y; ni++)
            {
                deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 0, ni);
            }
        }
    }

    CUDA_DEVICE_INLINE
    void run_nstage_v3(
        // SmemTensor3D<TypeA, Stages, PhysicalShapeAX, PhysicalShapeAY> smemA,
        SmemTensor4D<TypeA, Stages, BlockTileM / 8, BlockTileK / 8, 32> smemA,
        SmemTensor3D<TypeB, Stages, PhysicalShapeBX, PhysicalShapeBY> smemB,
        SmemTensor1D<half, CodeSize> smemCode,
        SmemTensor1D<half, AbsMaxPerBlock> smemAbsMax,
        Data4B accumulator[WarpMmaCountM][WarpMmaCountN][accBytes],
        int &useless)
    {

        constexpr Coord3D mmaSize = {MmaSizeM, MmaSizeN, MmaSizeK};
        constexpr Coord3D mmaMatrixCount = {mmaSize.x / 8, mmaSize.y / 8, mmaSize.z / 8};
        constexpr Coord3D warpMmaCount = {WarpMmaCountM, WarpMmaCountN, WarpMmaCountK}; // [4, 8, 2]
        constexpr Coord2D warpCount = {WarpCountM, WarpCountN};
        Coord2D warpIdx = {_wid / warpCount.y, _wid % warpCount.y};
        Coord2D laneIdx = {_lid / 8, _lid % 8};

        Data1B bufferA[warpMmaCount.z][warpMmaCount.x][4];
        Data4B bufferB[warpMmaCount.z][warpMmaCount.y][mmaMatrixCount.y * mmaMatrixCount.z / (QuantStatB == QUANTIZED ? 2 : 1)];
        // Data4B bufferB[][];

        Data4B fragA[warpMmaCount.z][warpMmaCount.x][mmaMatrixCount.x * mmaMatrixCount.z];
        Data4B fragB[warpMmaCount.z][warpMmaCount.y][mmaMatrixCount.y * mmaMatrixCount.z];

        const Coord2D logicalBaseIdxA = logical_base_idx_A();
        const Coord2D logicalBaseIdxB = logical_base_idx_B();

#pragma unroll
        for (int s = 0; s < Stages - 1; s++)
        {
            // TODO: LDGSTS LOGIC HERE
            deq4_ldgsts_A(smemA.get_child(s));
            deq4_ldgsts_B(smemB.get_child(s));
            deq4_ldgsts_move_to_next_tile();
            __pipeline_commit();
        }
        __pipeline_wait_prior(Stages - 3);
        __syncthreads();

        auto currentStageSmemPtrA = smemA.get_child(0);
        TypeB *currentStageSmemPtrB = smemB.get_child(0).startPtr;

#pragma unroll
        for (int i = 0; i < warpMmaCount.x; i++)
        {
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 1, i);
        }
#pragma unroll
        for (int i = 0; i < warpMmaCount.y / 2; i++)
        {
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 1, i);
        }
#pragma unroll
        for (int mi = 0; mi < warpMmaCount.x; mi++)
        {
            deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 0, mi);
            deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 1, mi);
        }
#pragma unroll
        for (int ni = 0; ni < warpMmaCount.y; ni++)
        {
            deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 0, ni);
            deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 1, ni);
        }

        currentStageSmemPtrA = smemA.get_child(1);
        currentStageSmemPtrB = smemB.get_child(1).startPtr;
#pragma unroll
        for (int i = 0; i < warpMmaCount.x; i++)
        {
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
        }
#pragma unroll
        for (int i = 0; i < warpMmaCount.y / 2; i++)
        {
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
        }

        for (int ki = 0; ki < _nMainIter; ki++)
        {
            int currentStage = ki % Stages;

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < warpMmaCount.y; ni++)
                {
                    // HMMA(mi, ni, 0);
                    deq4_hmma(fragA, fragB, accumulator, mi, ni, 0);
                }
            }

#pragma unroll
            for (int i = 0; i < warpMmaCount.x; i++)
            {
                deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 1, i);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.y / 2; i++)
            {
                deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 1, i);
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
                deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 0, mi);
            }
#pragma unroll
            for (int ni = 0; ni < warpMmaCount.y; ni++)
            {
                deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 0, ni);
            }

            if (ki < _nMainIter - (Stages - 1))
            {
                // TODO: LDGSTS logic here
                deq4_ldgsts_A(smemA.get_child((currentStage + Stages - 1) % Stages));
                deq4_ldgsts_B(smemB.get_child((currentStage + Stages - 1) % Stages));
                deq4_ldgsts_move_to_next_tile();
                __pipeline_commit();
            }
            __pipeline_wait_prior(Stages - 3);
            __syncthreads();
            currentStageSmemPtrA = smemA.get_child((currentStage + 2) % Stages);
            currentStageSmemPtrB = smemB.get_child((currentStage + 2) % Stages).startPtr;

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < warpMmaCount.y; ni++)
                {
                    deq4_hmma(fragA, fragB, accumulator, mi, ni, 1);
                }
            }

#pragma unroll
            for (int i = 0; i < warpMmaCount.x; i++)
            {
                deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.y / 2; i++)
            {
                deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
                deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 1, mi);
            }

#pragma unroll
            for (int ni = 0; ni < warpMmaCount.y; ni++)
            {
                deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 1, ni);
            }
        }
    }

    CUDA_DEVICE_INLINE
    void run_nstage_v4(
        // SmemTensor3D<TypeA, Stages, PhysicalShapeAX, PhysicalShapeAY> smemA,
        SmemTensor4D<TypeA, Stages, BlockTileM / 8, BlockTileK / 8, 32> smemA,
        SmemTensor3D<TypeB, Stages, PhysicalShapeBX, PhysicalShapeBY> smemB,
        SmemTensor1D<half, CodeSize> smemCode,
        SmemTensor1D<half, AbsMaxPerBlock> smemAbsMax,
        Data4B accumulator[WarpMmaCountM][WarpMmaCountN][accBytes],
        int &useless)
    {
        // const int tid = threadIdx.x;
        // const int lid = tid % 32;
        // const int wid = __shfl_sync(0xffffffff, tid / 32, 0);
        constexpr Coord3D mmaSize = {MmaSizeM, MmaSizeN, MmaSizeK};
        constexpr Coord3D mmaMatrixCount = {mmaSize.x / 8, mmaSize.y / 8, mmaSize.z / 8};
        constexpr Coord3D warpMmaCount = {WarpMmaCountM, WarpMmaCountN, WarpMmaCountK}; // [4, 8, 2]
        constexpr Coord2D warpCount = {WarpCountM, WarpCountN};
        Coord2D warpIdx = {_wid / warpCount.y, _wid % warpCount.y};
        Coord2D laneIdx = {_lid / 8, _lid % 8};

        Data1B bufferA[warpMmaCount.z][warpMmaCount.x][4];
        Data4B bufferB[warpMmaCount.z][warpMmaCount.y][mmaMatrixCount.y * mmaMatrixCount.z / (QuantStatB == QUANTIZED ? 2 : 1)];

        Data4B fragA[warpMmaCount.z][warpMmaCount.x][mmaMatrixCount.x * mmaMatrixCount.z];
        Data4B fragB[warpMmaCount.z][warpMmaCount.y][mmaMatrixCount.y * mmaMatrixCount.z];

        const Coord2D logicalBaseIdxA = logical_base_idx_A();
        const Coord2D logicalBaseIdxB = logical_base_idx_B();

#pragma unroll
        for (int s = 0; s < Stages - 1; s++)
        {
            // TODO: LDGSTS LOGIC HERE
            deq4_ldgsts_A(smemA.get_child(s));
            deq4_ldgsts_B(smemB.get_child(s));
            deq4_ldgsts_move_to_next_tile();
            __pipeline_commit();
        }
        __pipeline_wait_prior(Stages - 3);
        __syncthreads();

        auto currentStageSmemPtrA = smemA.get_child(0);
        TypeB *currentStageSmemPtrB = smemB.get_child(0).startPtr;

#pragma unroll
        for (int i = 0; i < warpMmaCount.x; i++)
        {
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 1, i);
        }
#pragma unroll
        for (int i = 0; i < warpMmaCount.y / 2; i++)
        {
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 1, i);
        }
#pragma unroll
        for (int mi = 0; mi < warpMmaCount.x; mi++)
        {
            deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 0, mi);
        }
#pragma unroll
        for (int ni = 0; ni < warpMmaCount.y; ni++)
        {
            deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 0, ni);
        }

        currentStageSmemPtrA = smemA.get_child(1);
        currentStageSmemPtrB = smemB.get_child(1).startPtr;
#pragma unroll
        for (int i = 0; i < warpMmaCount.x; i++)
        {
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
        }
#pragma unroll
        for (int i = 0; i < warpMmaCount.y / 2; i++)
        {
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
        }

        for (int ki = 0; ki < _nMainIter; ki++)
        {
            int currentStage = ki % Stages;

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < warpMmaCount.y; ni++)
                {
                    // HMMA(mi, ni, 0);
                    deq4_hmma(fragA, fragB, accumulator, mi, ni, 0);
                }
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
                deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 1, mi);
            }
#pragma unroll
            for (int ni = 0; ni < warpMmaCount.y; ni++)
            {
                deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 1, ni);
            }

#pragma unroll
            for (int i = 0; i < warpMmaCount.x; i++)
            {
                deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 1, i);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.y / 2; i++)
            {
                deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 1, i);
            }

            if (ki < _nMainIter - (Stages - 1))
            {
                // TODO: LDGSTS logic here
                deq4_ldgsts_A(smemA.get_child((currentStage + Stages - 1) % Stages));
                deq4_ldgsts_B(smemB.get_child((currentStage + Stages - 1) % Stages));
                deq4_ldgsts_move_to_next_tile();
                __pipeline_commit();
            }
            __pipeline_wait_prior(Stages - 3);
            __syncthreads();
            currentStageSmemPtrA = smemA.get_child((currentStage + 2) % Stages);
            currentStageSmemPtrB = smemB.get_child((currentStage + 2) % Stages).startPtr;

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < warpMmaCount.y; ni++)
                {
                    deq4_hmma(fragA, fragB, accumulator, mi, ni, 1);
                }
            }
#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
                deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 0, mi);
            }

#pragma unroll
            for (int ni = 0; ni < warpMmaCount.y; ni++)
            {
                deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 0, ni);
            }

#pragma unroll
            for (int i = 0; i < warpMmaCount.x; i++)
            {
                deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.y / 2; i++)
            {
                deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
            }
        }
    }

    CUDA_DEVICE_INLINE
    void run_nstage_v5(
        // SmemTensor3D<TypeA, Stages, PhysicalShapeAX, PhysicalShapeAY> smemA,
        SmemTensor4D<TypeA, Stages, BlockTileM / 8, BlockTileK / 8, 32> smemA,
        SmemTensor3D<TypeB, Stages, PhysicalShapeBX, PhysicalShapeBY> smemB,
        SmemTensor1D<half, CodeSize> smemCode,
        SmemTensor1D<half, AbsMaxPerBlock> smemAbsMax,
        Data4B accumulator[WarpMmaCountM][WarpMmaCountN][accBytes],
        int &useless)
    {
        // const int tid = threadIdx.x;
        // const int lid = tid % 32;
        // const int wid = __shfl_sync(0xffffffff, tid / 32, 0);
        constexpr Coord3D mmaSize = {MmaSizeM, MmaSizeN, MmaSizeK};
        constexpr Coord3D mmaMatrixCount = {mmaSize.x / 8, mmaSize.y / 8, mmaSize.z / 8};
        constexpr Coord3D warpMmaCount = {WarpMmaCountM, WarpMmaCountN, WarpMmaCountK}; // [4, 8, 2]
        constexpr Coord2D warpCount = {WarpCountM, WarpCountN};
        Coord2D warpIdx = {_wid / warpCount.y, _wid % warpCount.y};
        Coord2D laneIdx = {_lid / 8, _lid % 8};

        Data1B bufferA[warpMmaCount.z][warpMmaCount.x][4];
        Data4B bufferB[warpMmaCount.z][warpMmaCount.y][mmaMatrixCount.y * mmaMatrixCount.z / (QuantStatB == QUANTIZED ? 2 : 1)];
        // Data4B bufferB[][];

        Data4B fragA[warpMmaCount.z][warpMmaCount.x][mmaMatrixCount.x * mmaMatrixCount.z];
        Data4B fragB[warpMmaCount.z][warpMmaCount.y][mmaMatrixCount.y * mmaMatrixCount.z];

        const Coord2D logicalBaseIdxA = logical_base_idx_A();
        const Coord2D logicalBaseIdxB = logical_base_idx_B();

#pragma unroll
        for (int s = 0; s < Stages - 1; s++)
        {
            // TODO: LDGSTS LOGIC HERE
            deq4_ldgsts_A(smemA.get_child(s));
            deq4_ldgsts_B(smemB.get_child(s));
            deq4_ldgsts_move_to_next_tile();
            __pipeline_commit();
        }
        __pipeline_wait_prior(Stages - 3);
        __syncthreads();

        auto currentStageSmemPtrA = smemA.get_child(0);
        TypeB *currentStageSmemPtrB = smemB.get_child(0).startPtr;

#pragma unroll
        for (int i = 0; i < warpMmaCount.x; i++)
        {
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 1, i);
        }
#pragma unroll
        for (int i = 0; i < warpMmaCount.y / 2; i++)
        {
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 1, i);
        }
#pragma unroll
        for (int mi = 0; mi < warpMmaCount.x; mi++)
        {
            deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 0, mi);
            deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 1, mi);
        }
#pragma unroll
        for (int ni = 0; ni < warpMmaCount.y; ni++)
        {
            deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 0, ni);
            deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 1, ni);
        }

        currentStageSmemPtrA = smemA.get_child(1);
        currentStageSmemPtrB = smemB.get_child(1).startPtr;
#pragma unroll
        for (int i = 0; i < warpMmaCount.x; i++)
        {
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
            deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 1, i);
        }
#pragma unroll
        for (int i = 0; i < warpMmaCount.y / 2; i++)
        {
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
            deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 1, i);
        }

        for (int ki = 0; ki < _nMainIter; ki++)
        {
            int currentStage = ki % Stages;
            currentStageSmemPtrA = smemA.get_child((currentStage + 2) % Stages);
            currentStageSmemPtrB = smemB.get_child((currentStage + 2) % Stages).startPtr;
            __pipeline_wait_prior(Stages - 3);
            __syncthreads();

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < warpMmaCount.y; ni++)
                {
                    // HMMA(mi, ni, 0);
                    deq4_hmma(fragA, fragB, accumulator, mi, ni, 0);
                }
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
                deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 0, mi);
            }
#pragma unroll
            for (int ni = 0; ni < warpMmaCount.y; ni++)
            {
                deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 0, ni);
            }

#pragma unroll
            for (int i = 0; i < warpMmaCount.x; i++)
            {
                deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 0, i);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.y / 2; i++)
            {
                deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 0, i);
            }

            if (ki < _nMainIter - (Stages - 1))
            {
                // TODO: LDGSTS logic here
                deq4_ldgsts_A(smemA.get_child((currentStage + Stages - 1) % Stages));
                deq4_ldgsts_B(smemB.get_child((currentStage + Stages - 1) % Stages));
                deq4_ldgsts_move_to_next_tile();
                __pipeline_commit();
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
#pragma unroll
                for (int ni = 0; ni < warpMmaCount.y; ni++)
                {
                    deq4_hmma(fragA, fragB, accumulator, mi, ni, 1);
                }
            }

#pragma unroll
            for (int mi = 0; mi < warpMmaCount.x; mi++)
            {
                deq4_transform_frag_A(bufferA, fragA, smemCode, smemAbsMax, 1, mi);
            }

#pragma unroll
            for (int ni = 0; ni < warpMmaCount.y; ni++)
            {
                deq4_transform_frag_B(bufferB, fragB, smemCode, smemAbsMax, 1, ni);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.x; i++)
            {
                deq4_ldsm_A(currentStageSmemPtrA, bufferA, logicalBaseIdxA, 1, i);
            }
#pragma unroll
            for (int i = 0; i < warpMmaCount.y / 2; i++)
            {
                deq4_ldsm_B(currentStageSmemPtrB, bufferB, logicalBaseIdxB, 1, i);
            }
        }
    }

    CUDA_DEVICE_INLINE
    void run(
        SmemTensor4D<TypeA, Stages, BlockTileM / 8, BlockTileK / 8, 32> smemA,
        SmemTensor3D<TypeB, Stages, PhysicalShapeBX, PhysicalShapeBY> smemB,
        SmemTensor1D<half, CodeSize> smemCode,
        SmemTensor1D<half, AbsMaxPerBlock> smemAbsMax,
        Data4B accumulator[WarpMmaCountM][WarpMmaCountN][accBytes],
        int &useless)
    {
        switch (PipelineStrat)
        {
        case 1:
            run_nstage(smemA, smemB, smemCode, smemAbsMax, accumulator, useless);
            break;

        case 2:
            run_nstage_v2(smemA, smemB, smemCode, smemAbsMax, accumulator, useless);
            break;

        case 3:
            run_nstage_v3(smemA, smemB, smemCode, smemAbsMax, accumulator, useless);
            break;

        case 4:
            run_nstage_v4(smemA, smemB, smemCode, smemAbsMax, accumulator, useless);
            break;

        case 5:
            run_nstage_v5(smemA, smemB, smemCode, smemAbsMax, accumulator, useless);
            break;
        }
        // run_nstage_v2(smemA, smemB, smemCode, smemAbsMax, accumulator, useless);
        // run_nstage_v5(smemA, smemB, smemCode, smemAbsMax, accumulator, useless);
    }

    CUDA_DEVICE_INLINE
    void deq4_ldgsts_move_to_next_tile()
    {
        _globalStart.z += BlockTileK;
        if (_globalStart.z + BlockTileK > _globalEnd.z)
        {
            if constexpr (TransformA == TRANSFORM_N)
            {
                _validTileSizeA.y = _globalEnd.z - _globalStart.z;
            }
            else
            {
                _validTileSizeA.x = _globalEnd.z - _globalStart.z;
            }

            if constexpr (TransformB == TRANSFORM_N)
            {
                _validTileSizeB.x = _globalEnd.z - _globalStart.z;
            }
            else
            {
                _validTileSizeB.y = _globalEnd.z - _globalStart.z;
            }
        }
    }

    CUDA_DEVICE_INLINE
    void deq4_ldgsts_A(
        SmemTensor3D<TypeA, BlockTileM / 8, BlockTileK / 8, 32> smemA)
    {
        TypeA *smemPtrA = smemA.startPtr;
        // static_assert(TransformA == TRANSFORM_N);
        constexpr int warpCount = WarpCountM * WarpCountN;
        // each thread loads alignSizeBytes
        // each 32 / alignSizeBytes threads loads 32 bytes
        // each warp loads alignSizeBytes of 32 bytes
        // each thread block loads alignSizeBytes * numWarps of 32 bytes at max
        //
        Coord2D warpShape = {align_size_bytes_A(), 32 / align_size_bytes_A()}; // {4, 8} or {8, 4} or {16, 2}
        Coord2D loadIters = {div_ru((BlockTileM / 8), (warpShape.x * warpCount)), BlockTileK / 8};
        Coord2D laneIdx = {_lid / warpShape.y, _lid % warpShape.y};
        // TileM | TileK | alignSize | warpCount | loadIters | warpShape |     laneIdx     |
        // 1024  | 32    | 4         | 8         | {4, 4}    | {4, 8}    | {0...4, 0...8}  |
        // 1024  | 32    | 8         | 8         | {2, 4}    | {8, 4}    | {0...8, 0...4}  |
        // 1024  | 32    | 16        | 8         | {1, 4}    | {16, 2}   | {0...16, 0...2} |

#pragma unroll
        for (int i = 0; i < loadIters.x; i++)
        {
#pragma unroll
            for (int j = 0; j < loadIters.y; j++)
            {
                Coord3D localIndex = {
                    i * (warpShape.x * warpCount) + _wid * warpShape.x + laneIdx.x,
                    j,
                    laneIdx.y * align_size_bytes_A()};
                Coord3D globalIndex = {_globalStart.x / 8 + localIndex.x, _globalStart.z / 8 + localIndex.y, localIndex.z};
                Coord2D strideA = {(_strideA.x / 8) * 32, 32};
                long long globalOffset = globalIndex.x * strideA.x + globalIndex.y * strideA.y + globalIndex.z;
                int zfill = 0;
                const TypeA *ptrA = reinterpret_cast<const TypeA *>(_ptrA);
                if (localIndex.x < (BlockTileM / 8) && globalIndex.x < (_problemSize.x / 8))
                {
                    __pipeline_memcpy_async(
                        smemA.get_ptr(localIndex.x, localIndex.y, localIndex.z),
                        &ptrA[globalOffset],
                        align_size_bytes_A(),
                        zfill);
                }
            }
        }
    }

    CUDA_DEVICE_INLINE
    void deq4_ldgsts_B(
        SmemTensor2D<TypeB, PhysicalShapeBX, PhysicalShapeBY> smemB)
    {
        TypeB *smemPtrB = smemB.startPtr;

        const Coord2D matrixShape = matrix_shape_B();
        const Coord2D globalStart = global_start_B();

#pragma unroll
        for (int i = 0; i < load_iters_B().x; i++)
        {
#pragma unroll
            for (int j = 0; j < load_iters_B().y; j++)
            {
                Coord2D threadStart = {
                    thread_base_start_B().x + i * max_tb_load_shape_B().x,
                    thread_base_start_B().y + j * max_tb_load_shape_B().y};

                Coord2D threadEnd = {threadStart.x, threadStart.y + AlignSizeB};
                Coord2D threadGlobalStart = {globalStart.x + threadStart.x, globalStart.y + threadStart.y};
                Coord2D threadGlobalEnd = {threadGlobalStart.x, threadGlobalStart.y + AlignSizeB};

                int globalOffset = threadGlobalStart.x * _strideB.x + threadGlobalStart.y;
                int sharedOffset = _converterB.convert_to_offset(threadStart);
                int zfill = 0;

                if constexpr (true)
                {
                    if (likely(threadGlobalEnd.x < matrixShape.x && threadGlobalEnd.y < matrixShape.y))
                    {
                        zfill = 0;
                    }
                    else if (threadGlobalEnd.x < matrixShape.x && threadGlobalStart.y < matrixShape.y && threadGlobalEnd.y >= matrixShape.y)
                    {
                        zfill = matrixShape.y - threadGlobalEnd.y;
                    }
                    else
                    {
                        zfill = align_size_bytes_B();
                    }
                }

                const TypeB *ptrB = reinterpret_cast<const TypeB *>(_ptrB);

                if (threadStart.x < _validTileSizeB.x && threadStart.y < _validTileSizeB.y)
                {
                    __pipeline_memcpy_async(
                        &smemPtrB[sharedOffset],
                        &ptrB[globalOffset],
                        align_size_bytes_B(),
                        zfill);
                }
            }
        }
    }

    CUDA_DEVICE_INLINE
    void deq4_ldsm_A(
        // TypeA* smemPtrA,
        SmemTensor3D<TypeA, BlockTileM / 8, BlockTileK / 8, 32> smemA,
        Data1B bufferA[WarpMmaCountK][WarpMmaCountM][4],
        Coord2D logicalBaseIdxA,
        int ki,
        int mi)
    {
// FIXME: Maybe does not work
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
#pragma unroll
            for (int j = 0; j < 2; j++)
            {
                bufferA[ki][mi][j * 2 + i].as_uint8[0] = smemA.get(
                    _warpIdx.x * 2 * WarpMmaCountM + mi * 2 + i,
                    ki * 2 + j,
                    _lid);
            }
        }
    }

    CUDA_DEVICE_INLINE
    void deq4_ldsm_B(
        TypeB *smemPtrB,
        Data4B bufferB[WarpMmaCountK][WarpMmaCountN][(MmaSizeN / 8) * (MmaSizeK / 8) / (QuantStatB == QUANTIZED ? 2 : 1)],
        Coord2D logicalBaseIdxB,
        int ki,
        int ni)
    {
        constexpr Coord3D mmaMatrixCount = {MmaSizeM / 8, MmaSizeN / 8, MmaSizeK / 8};
        if (TransformB == TRANSFORM_N)
        {

            Data4B temp[4];
            Coord2D logicalIdxB = {
                logicalBaseIdxB.x + ki * mmaMatrixCount.z * 8,
                logicalBaseIdxB.y + ni * mmaMatrixCount.y * 2 * 8};
            ldsm<4, layout_B()>(smemPtrB + _converterB.convert_to_offset(logicalIdxB), temp);
            bufferB[ki][ni * 2 + 0][0].as_half2[0] = temp[0].as_half2[0];
            bufferB[ki][ni * 2 + 0][1].as_half2[0] = temp[1].as_half2[0];
            bufferB[ki][ni * 2 + 1][0].as_half2[0] = temp[2].as_half2[0];
            bufferB[ki][ni * 2 + 1][1].as_half2[0] = temp[3].as_half2[0];
        }
        else if (TransformB == TRANSFORM_T)
        {

            Data4B temp[4];
            Coord2D logicalIdxB = {
                logicalBaseIdxB.x + ni * mmaMatrixCount.y * 2 * 8,
                logicalBaseIdxB.y + ki * mmaMatrixCount.z * 8};
            ldsm<4, layout_B()>(smemPtrB + _converterB.convert_to_offset(logicalIdxB), temp);
            bufferB[ki][ni * 2 + 0][0].as_half2[0] = temp[0].as_half2[0];
            bufferB[ki][ni * 2 + 0][1].as_half2[0] = temp[1].as_half2[0];
            bufferB[ki][ni * 2 + 1][0].as_half2[0] = temp[2].as_half2[0];
            bufferB[ki][ni * 2 + 1][1].as_half2[0] = temp[3].as_half2[0];
        }
    }

    CUDA_DEVICE_INLINE
    void deq4_transform_frag_A(
        Data1B bufferA[WarpMmaCountK][WarpMmaCountM][4],
        Data4B fragA[WarpMmaCountK][WarpMmaCountM][(MmaSizeM / 8) * (MmaSizeK / 8)],
        // 4 x (2 int4) --> 4 x (2 half)
        //
        //  [[r00, r01]      [(R00, R01), ..., (R20, R21), ...]   [(r00, r01), ..., (r02, r03), ...]
        //   [r10, r11]  --> [................................] = [................................]
        //   [r20, r21]      [(R10, R11), ..., (R30, R31), ...]   [(r10, r11), ..., (r12, r13), ...]
        //   [r30, r31]]     [................................]   [................................]
        SmemTensor1D<half, CodeSize> smemCode,
        SmemTensor1D<half, AbsMaxPerBlock> smemAbsMax,
        int ki, int mi)
    {
        constexpr Coord3D mmaMatrixCount = {MmaSizeM / 8, MmaSizeN / 8, MmaSizeK / 8};
        // static_assert(QuantStatA == QUANTIZED);
        int temp = _warpLogicalIdxStart.x * 8 + _laneIdx4.x + mi * mmaMatrixCount.x * 8;
        half absMax1 = smemAbsMax.get(temp);
        half absMax2 = smemAbsMax.get(temp + 8);
        constexpr uint8_t mask = 0b00001111;
        fragA[ki][mi][0].as_half[0] = smemCode.get(bufferA[ki][mi][0].as_uint8[0] >> 4 & mask) * absMax1;
        fragA[ki][mi][0].as_half[1] = smemCode.get(bufferA[ki][mi][0].as_uint8[0] & mask) * absMax1;

        fragA[ki][mi][1].as_half[0] = smemCode.get(bufferA[ki][mi][1].as_uint8[0] >> 4 & mask) * absMax2;
        fragA[ki][mi][1].as_half[1] = smemCode.get(bufferA[ki][mi][1].as_uint8[0] & mask) * absMax2;

        fragA[ki][mi][2].as_half[0] = smemCode.get(bufferA[ki][mi][2].as_uint8[0] >> 4 & mask) * absMax1;
        fragA[ki][mi][2].as_half[1] = smemCode.get(bufferA[ki][mi][2].as_uint8[0] & mask) * absMax1;

        fragA[ki][mi][3].as_half[0] = smemCode.get(bufferA[ki][mi][3].as_uint8[0] >> 4 & mask) * absMax2;
        fragA[ki][mi][3].as_half[1] = smemCode.get(bufferA[ki][mi][3].as_uint8[0] & mask) * absMax2;
    }

    CUDA_DEVICE_INLINE
    void deq4_transform_frag_B(
        Data4B bufferB[WarpMmaCountK][WarpMmaCountN][(MmaSizeN / 8) * (MmaSizeK / 8) / (QuantStatB == QUANTIZED ? 2 : 1)],
        Data4B fragB[WarpMmaCountK][WarpMmaCountN][(MmaSizeN / 8) * (MmaSizeK / 8)],
        SmemTensor1D<half, CodeSize> smemCode,
        SmemTensor1D<half, AbsMaxPerBlock> smemAbsMax,
        int ki, int ni)
    {
        constexpr Coord3D mmaMatrixCount = {MmaSizeM / 8, MmaSizeN / 8, MmaSizeK / 8};
        // static_assert(QuantStatB == NOT_QUANTIZED);
        if (QuantStatB == QUANTIZED)
        {
        }
        else
        {
// TODO: This should be a no-op if compiler is smart enough, if not, try swapping the pointers of 2 register arrays.
#pragma unroll
            for (int i = 0; i < mmaMatrixCount.y * mmaMatrixCount.z; i++)
            {
                fragB[ki][ni][i].as_int32[0] = bufferB[ki][ni][i].as_int32[0];
            }
        }
    }

    CUDA_DEVICE_INLINE
    void deq4_hmma(
        Data4B fragA[WarpMmaCountK][WarpMmaCountM][(MmaSizeM / 8) * (MmaSizeK / 8)],
        Data4B fragB[WarpMmaCountK][WarpMmaCountN][(MmaSizeN / 8) * (MmaSizeK / 8)],
        Data4B accumulator[WarpMmaCountM][WarpMmaCountN][accBytes],
        int mi, int ni, int ki)
    {
        mma<MmaSizeM, MmaSizeN, MmaSizeK, LAYOUT_C, LAYOUT_F, AccMode>(fragA[ki][mi], fragB[ki][ni], accumulator[mi][ni]);
    }

    CUDA_DEVICE_INLINE
    Coord2D logical_base_idx_A()
    {
        Coord2D logicalBaseIdxA;

        logicalBaseIdxA = {
            (_warpLogicalIdxStart.x + _laneIdx8r02.y) * 8 + _laneIdx8.y,
            (0 + _laneIdx8r02.x) * 8};

        return logicalBaseIdxA;
    }

    CUDA_DEVICE_INLINE
    Coord2D logical_base_idx_B()
    {
        Coord2D logicalBaseIdxB;
        if (TransformB == TRANSFORM_N)
        {
            logicalBaseIdxB = {
                (_laneIdx8r02.y) * 8 + _laneIdx8.y,
                (_warpLogicalIdxStart.y + _laneIdx8r02.x * MmaSizeN / 8) * 8};
        }
        else
        {
            logicalBaseIdxB = {
                (_warpLogicalIdxStart.y + _laneIdx8r02.x * MmaSizeN / 8) * 8 + _laneIdx8.y,
                (_laneIdx8r02.y) * 8};
        }
        return logicalBaseIdxB;
    }

    CUDA_DEVICE_INLINE
    Coord2D matrix_shape_A()
    {
        if constexpr (TransformA == TRANSFORM_N)
        {
            return {_problemSize.x, _problemSize.z};
        }
        else
        {
            return {_problemSize.z, _problemSize.x};
        }
    }

    CUDA_DEVICE_INLINE
    Coord2D matrix_shape_B()
    {
        if constexpr (TransformB == TRANSFORM_N)
        {
            return {_problemSize.z, _problemSize.y};
        }
        else
        {
            return {_problemSize.y, _problemSize.z};
        }
    }

    CUDA_DEVICE_INLINE
    Coord2D global_start_A()
    {
        if constexpr (TransformA == TRANSFORM_N)
        {
            return {_globalStart.x, _globalStart.z};
        }
        else
        {
            return {_globalStart.z, _globalStart.x};
        }
    }

    CUDA_DEVICE_INLINE
    Coord2D global_start_B()
    {
        if constexpr (TransformB == TRANSFORM_N)
        {
            return {_globalStart.z, _globalStart.y};
        }
        else
        {
            return {_globalStart.y, _globalStart.z};
        }
    }

    CUDA_DEVICE_INLINE
    Coord2D global_end_A()
    {
        if constexpr (TransformA == TRANSFORM_N)
        {
            return {_globalEnd.x, _globalEnd.z};
        }
        else
        {
            return {_globalEnd.z, _globalEnd.x};
        }
    }

    CUDA_DEVICE_INLINE
    Coord2D global_end_B()
    {
        if constexpr (TransformB == TRANSFORM_N)
        {
            return {_globalEnd.z, _globalEnd.y};
        }
        else
        {
            return {_globalEnd.y, _globalEnd.z};
        }
    }

    CUDA_DEVICE_INLINE
    Coord2D thread_base_start_A()
    {
        Coord2D result;

        result = {
            _wid * max_warp_load_shape_A().x + _laneIdxA.x,
            _laneIdxA.y * AlignSizeA};

        return result;
    }

    CUDA_DEVICE_INLINE
    Coord2D thread_base_start_B()
    {
        Coord2D result;

        result = {
            _wid * max_warp_load_shape_B().x + _laneIdxB.x,
            _laneIdxB.y * AlignSizeB};

        return result;
    }

private:
    const int _nMainIter;
    const void *_ptrA;
    const void *_ptrB;
    const int _wid;
    const int _lid;
    const int _tid;
    Coord1D _strideA;
    Coord1D _strideB;
    Coord3D _problemSize;
    Coord3D _globalStart;
    Coord3D _globalEnd;
    Coord3D _leadingDims;
    Coord2D _laneIdx8;
    Coord2D _laneIdx8r02;
    Coord2D _laneIdx4;
    Coord2D _warpIdx;
    Coord2D _warpLogicalIdxStart;
    Coord2D _laneIdxA;
    Coord2D _laneIdxB;
    Coord2D _validTileSizeA;
    Coord2D _validTileSizeB;

    CUDA_DEVICE_INLINE
    static constexpr MATRIX_LAYOUT layout_A()
    {
        return TransformA == TRANSFORM_N ? LAYOUT_C : LAYOUT_F;
    }

    CUDA_DEVICE_INLINE
    static constexpr MATRIX_LAYOUT layout_B()
    {
        return TransformB == TRANSFORM_N ? LAYOUT_F : LAYOUT_C;
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D logical_tile_size_A()
    {
        if constexpr (TransformA == TRANSFORM_N)
        {
            return {BlockTileM, BlockTileK};
        }
        else
        {
            return {BlockTileK, BlockTileM};
        }
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D logical_tile_size_B()
    {
        if constexpr (TransformB == TRANSFORM_N)
        {
            return {BlockTileK, BlockTileN};
        }
        else
        {
            return {BlockTileN, BlockTileK};
        }
    }

    CUDA_DEVICE_INLINE
    static constexpr int align_size_bytes_A()
    {
        return AlignSizeA * sizeof(TypeA);
    }

    CUDA_DEVICE_INLINE
    static constexpr int align_size_bytes_B()
    {
        return AlignSizeB * sizeof(TypeB);
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D lane_shape_A()
    {
        return {32 / (ContiguousBytesA / align_size_bytes_A()), (ContiguousBytesA / align_size_bytes_A())};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D lane_shape_B()
    {
        return {32 / (ContiguousBytesB / align_size_bytes_B()), (ContiguousBytesB / align_size_bytes_B())};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D max_warp_load_shape_A()
    {
        return {lane_shape_A().x, lane_shape_A().y * AlignSizeA};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D max_warp_load_shape_B()
    {
        return {lane_shape_B().x, lane_shape_B().y * AlignSizeB};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D max_tb_load_shape_A()
    {
        Coord2D temp1 = {max_warp_load_shape_A().x * WarpCountM * WarpCountN, max_warp_load_shape_A().y};
        Coord2D temp2 = {max_warp_load_shape_A().x, max_warp_load_shape_A().y * WarpCountM * WarpCountN};
        return (TransformA == TRANSFORM_N) ? temp1 : temp2;
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D max_tb_load_shape_B()
    {
        Coord2D temp1 = {max_warp_load_shape_B().x, max_warp_load_shape_B().y * WarpCountM * WarpCountN};
        Coord2D temp2 = {max_warp_load_shape_B().x * WarpCountM * WarpCountN, max_warp_load_shape_B().y};
        return (TransformB == TRANSFORM_N) ? temp1 : temp2;
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D load_iters_A()
    {
        return {
            div_ru(logical_tile_size_A().x, max_tb_load_shape_A().x),
            div_ru(logical_tile_size_A().y, max_tb_load_shape_A().y)};
    }

    CUDA_DEVICE_INLINE
    static constexpr Coord2D load_iters_B()
    {
        return {
            div_ru(logical_tile_size_B().x, max_tb_load_shape_B().x),
            div_ru(logical_tile_size_B().y, max_tb_load_shape_B().y)};
    }

    TileIndexConverter<
        TypeA,                   // uint8
        logical_tile_size_A().x, // 256
        logical_tile_size_A().y, // 32
        AlignSizeA               // 16
        >
        _converterA;

    TileIndexConverter<
        TypeB,
        logical_tile_size_B().x,
        logical_tile_size_B().y,
        AlignSizeB>
        _converterB;
};
template <
    int TileM,
    int TileN,
    int PatchM,
    int PatchN>
CUDA_DEVICE_INLINE
    Coord3D
    get_global_start(
        int n,
        int kPerSplit,
        int kid)
{
    int px = blockIdx.x % PatchN;
    int py = blockIdx.x / PatchN;
    int blockDimX = (n + (TileN * PatchN) - 1) / (TileN * PatchN);
    int blockIdxX = (blockIdx.y % blockDimX) * PatchN + px;
    int blockIdxY = (blockIdx.y / blockDimX) * PatchM + py;
    int gStartm = blockIdxY * TileM;
    int gStartn = blockIdxX * TileN; // starting index of block on N axis
    int gStartk = kid * kPerSplit;
    Coord3D result = {gStartm, gStartn, gStartk};
    return result;
}

template <int Dim1, int Dim2, int Dim3>
CUDA_DEVICE_INLINE void fill_accumulator(
    Data4B accumulator[Dim1][Dim2][Dim3],
    // Data4B value
    int value)
{
#pragma unroll
    for (int i = 0; i < Dim1; i++)
    {
#pragma unroll
        for (int j = 0; j < Dim2; j++)
        {
#pragma unroll
            for (int k = 0; k < Dim3; k++)
            {
                accumulator[i][j][k].as_int32[0] = value;
            }
        }
    }
}

template <
    int mmaSizeM,   // 16
    int mmaSizeN,   // 8
    int mmaSizeK,   // 16
    int warpCountM, // 4
    int warpCountN,
    int warpMmaCountM, // 4
    int warpMmaCountN, // 8
    int warpMmaCountK,
    int paddingC,
    int accMode>
CUDA_DEVICE_INLINE void store_C(
    Data4B accumulator[warpMmaCountM][warpMmaCountN][accBytes],
    half *dest,
    SmemTensor3D<half, warpCountM * warpCountN, 8, 32 + paddingC> smem,
    Coord2D globalStart,
    Coord2D destShape, // [m, n]
    Coord1D destStride // [n]
)
{
    int tid = threadIdx.x;
    int lid = tid % 32; // laneID
    int wid = tid / 32; // warpID

    constexpr Coord3D mmaSize = {mmaSizeM, mmaSizeN, mmaSizeK};
    constexpr Coord3D mmaMatrixCount = {mmaSize.x / 8, mmaSize.y / 8, mmaSize.z / 8};
    constexpr Coord2D warpCount = {warpCountM, warpCountN};
    constexpr Coord3D warpMmaCount = {warpMmaCountM, warpMmaCountN, warpMmaCountK};

    // x, y position of warp in a 2x4 warp grid
    Coord2D warpIdx = {wid / warpCount.y, wid % warpCount.y};
    // x, y position of thread in a 8x4 warp
    Coord2D laneIdx4 = {lid / 4, lid % 4};

// TODO: this only works if mmaSize.y == 8.
#pragma unroll
    for (int g = 0; g < mmaMatrixCount.x; g++)
    {
#pragma unroll
        for (int i = 0; i < warpMmaCount.x; i++)
        {
// each warp stores 8*32 fp16 in smem to make it contiguous
#pragma unroll
            for (int oj = 0; oj < warpMmaCount.y / 4; oj++)
            {
                // for (int oj=0; oj< div_ru(warpMmaCount.y, 4); oj++){
#pragma unroll
                for (int j = 0; j < 4; j++)
                {
                    // if (oj*4 + j < warpMmaCount.y){
#pragma unroll
                    for (int h = 0; h < 2; h++)
                    {
                        if constexpr (accMode == ACC_MODE_HALF)
                        {
                            smem.set(
                                wid,
                                laneIdx4.x,
                                j * 4 * 2 + laneIdx4.y * 2 + h, // every 4 threads in warp has 1 row (8 fp16) of C
                                accumulator[i][oj * 4 + j][g].as_half[h]);
                        }
                        else
                        {
                            smem.set(
                                wid,
                                laneIdx4.x,
                                j * 4 * 2 + laneIdx4.y * 2 + h, // every 4 threads in warp has 1 row (8 fp16) of C
                                (half)accumulator[i][oj * 4 + j][g * 2 + h].as_float[0]);
                        }
                    }
                    // }
                }
                __syncwarp();

// store to global memory
#pragma unroll
                for (int j = 0; j < 8; j++)
                {
                    Coord2D localIdx = {
                        warpIdx.x * warpMmaCount.x * mmaSize.x + i * mmaSize.x + g * 8 + j,
                        warpIdx.y * warpMmaCount.y * mmaSize.y + oj * 32 + lid};
                    Coord2D globalIdx = {
                        globalStart.x + localIdx.x,
                        globalStart.y + localIdx.y};
                    if (globalIdx.x < destShape.x && globalIdx.y < destShape.y)
                    {
                        atomicAdd(
                            &dest[globalIdx.x * destStride.x + globalIdx.y],
                            smem.get(wid, j, lid));

                        // dest[globalIdx.x * destStride.x + globalIdx.y] = smem.get(wid, j, lid);
                    }
                }
            }
        }
    }
}

template <
    int mmaSizeM,   // 16
    int mmaSizeN,   // 8
    int mmaSizeK,   // 16
    int warpCountM, // 4
    int warpCountN,
    int warpMmaCountM, // 4
    int warpMmaCountN, // 8
    int warpMmaCountK,
    int paddingC,
    int accMode,
    int splitK>
CUDA_DEVICE_INLINE void store_F(
    Data4B accumulator[warpMmaCountM][warpMmaCountN][accBytes],
    half *dest,
    SmemTensor3D<half, warpCountM * warpCountN, 8, 32 + paddingC> smem,
    Coord2D globalStart,
    Coord2D destShape, // [m, n]
    Coord1D destStride // [n])
)
{
    int tid = threadIdx.x;
    int lid = tid % 32; // laneID
    int wid = tid / 32; // warpID

    constexpr bool shouldAccumulate = splitK > 1;

    constexpr Coord3D mmaSize = {mmaSizeM, mmaSizeN, mmaSizeK};
    constexpr Coord3D mmaMatrixCount = {mmaSize.x / 8, mmaSize.y / 8, mmaSize.z / 8};
    constexpr Coord2D warpCount = {warpCountM, warpCountN};
    constexpr Coord3D warpMmaCount = {warpMmaCountM, warpMmaCountN, warpMmaCountK};

    // x, y position of warp in a 2x4 warp grid
    Coord2D warpIdx = {wid / warpCount.y, wid % warpCount.y};
    // x, y position of thread in a 8x4 warp
    Coord2D laneIdx4 = {lid / 4, lid % 4};

// TODO: this only works if mmaSize.y == 8.
#pragma unroll
    for (int i = 0; i < warpMmaCount.y; i++)
    {
#pragma unroll
        for (int oj = 0; oj < div_ru(warpMmaCount.x, 2); oj++)
        {
#pragma unroll
            for (int j = 0; j < 2; j++)
            {
                // if (oj*2+j < warpMmaCount.x){
                if (true)
                {
#pragma unroll
                    for (int g = 0; g < 2; g++)
                    {
#pragma unroll
                        for (int h = 0; h < 2; h++)
                        {
                            if constexpr (accMode == ACC_MODE_HALF)
                            {
                                smem.set(
                                    wid,
                                    2 * laneIdx4.y + h,
                                    j * 16 + g * 8 + laneIdx4.x,
                                    accumulator[oj * 2 + j][i][g].as_half[h]);
                            }
                            else
                            {
                                smem.set(
                                    wid,
                                    2 * laneIdx4.y + h,
                                    j * 16 + g * 8 + laneIdx4.x,
                                    (half)accumulator[oj * 2 + j][i][g * 2 + h].as_float[0]);
                            }
                        }
                    }
                }
            }

            __syncwarp();
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                Coord2D localIdx = {
                    warpIdx.y * warpMmaCount.y * mmaSize.y + i * mmaSize.y + j,
                    warpIdx.x * warpMmaCount.x * mmaSize.x + oj * 32 + lid};
                Coord2D globalIdx = {
                    globalStart.x + localIdx.x,
                    globalStart.y + localIdx.y};
                if (globalIdx.x < destShape.x && globalIdx.y < destShape.y)
                {

                    if constexpr (shouldAccumulate)
                    {

                        atomicAdd(
                            &dest[globalIdx.x * destStride.x + globalIdx.y],
                            smem.get(wid, j, lid));
                    }
                    else
                    {
                        dest[globalIdx.x * destStride.x + globalIdx.y] = smem.get(wid, j, lid);
                    }
                }
            }
        }
    }
}

template <
    typename T,
    int AbsMaxPerBlock,
    int ThreadsPerBlock>
CUDA_DEVICE_INLINE void load_abs_max(
    SmemTensor1D<T, AbsMaxPerBlock> smemAbsMax,
    const float *absMaxPtr,
    const int globalStart,
    const int tid)
{
    if (AbsMaxPerBlock <= ThreadsPerBlock)
    {
        if (tid < AbsMaxPerBlock)
        {
            smemAbsMax.set(tid, (T)absMaxPtr[globalStart + tid]);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < (AbsMaxPerBlock + ThreadsPerBlock - 1) / ThreadsPerBlock; i++)
        {
            int idx = i * ThreadsPerBlock + tid;
            if (idx < AbsMaxPerBlock)
            {
                smemAbsMax.set(idx, (T)absMaxPtr[globalStart + idx]);
            }
        }
    }
}

template <
    typename T,
    int CodeSize,
    int ThreadsPerBlock>
CUDA_DEVICE_INLINE void load_code_v2(
    SmemTensor1D<T, CodeSize> smemCode,
    const float *codePtr,
    const int tid)
{
    if (CodeSize <= ThreadsPerBlock)
    {
        if (tid < CodeSize)
        {
            smemCode.set(tid, (T)codePtr[tid]);
        }
    }
    else
    {
#pragma unroll
        for (int i = 0; i < (CodeSize + ThreadsPerBlock - 1) / ThreadsPerBlock; i++)
        {
            int idx = i * ThreadsPerBlock + tid;
            if (idx < CodeSize)
            {
                smemCode.set(idx, (T)codePtr[idx]);
            }
        }
    }
}
template <
    int alignSizeBytesA,
    int alignSizeBytesB,
    int tileM,
    int tileN,
    int tileK,
    int patchM,
    int patchN,
    int k,
    int m,
    int splitK,
    int warpCountM,
    int warpCountN,
    int mmaSizeM,
    int mmaSizeN,
    int mmaSizeK,
    int warpMmaCountM,
    int warpMmaCountN,
    int warpMmaCountK,
    int contiguousBytesA,
    int contiguousBytesB,
    int deqBlockSize,
    int stages,
    int absMaxPerBlock,
    int threadsPerBlock,
    int codeSize,
    bool isSafe,
    int pipelineStrat,
    int paddingC,
    MATRIX_TRANSFORM transformA,
    MATRIX_TRANSFORM transformB,
    MATRIX_TRANSFORM transformC,
    MATRIX_QUANTIZE_STATUS quantStatA,
    MATRIX_QUANTIZE_STATUS quantStatB>
__global__ void Tuna(
    const __restrict__ uint8_t *matA, //[b, m/8, k/8, 32]
    const __restrict__ half *matB,    //[n, k, n]
    const float *codePtr,             //[16]
    const float *absMaxPtr,           //[b, deqBlockCount]
    half __restrict__ *matC,          //[b, m, n]
    int n)
{
    const int tid = threadIdx.x;
    const int kid = blockIdx.z;

    constexpr int alignSizeA = alignSizeBytesA / sizeof(typeA);
    constexpr int alignSizeB = alignSizeBytesB / sizeof(typeB);

    constexpr int kPerSplit = div_ru(div_ru(k, tileK), splitK) * tileK;
    const Coord3D globalStart = get_global_start<tileM, tileN, patchM, patchN>(n, kPerSplit, kid);
    const Coord3D globalEnd = {globalStart.x + tileM, globalStart.y + tileN, min(globalStart.z + kPerSplit, k)};
    constexpr Coord2D warpCount = {warpCountM, warpCountN};
    constexpr Coord3D mmaSize = {mmaSizeM, mmaSizeN, mmaSizeK};
    constexpr Coord3D mmaMatrixCount = {mmaSize.x / 8, mmaSize.y / 8, mmaSize.z / 8};
    constexpr Coord3D tileSize = {tileM, tileN, tileK};
    constexpr Coord3D warpMmaCount = {warpMmaCountM, warpMmaCountN, warpMmaCountK};
    constexpr Coord3D leadingDims = {k, k, m};

    const Coord3D problemSize = {m, n, k};
    if (globalStart.x >= problemSize.x || globalStart.y >= problemSize.y)
    {
        return;
    }

    constexpr int deqBlockCount = m * k / deqBlockSize;

    TileIndexConverter<
        typeA,
        (tileSize.x / 8) * (tileSize.z / 8),
        32,
        alignSizeA>
        converterA;

    TileIndexConverter<
        typeB,
        tileSize.y,
        tileSize.z,
        alignSizeB>
        converterB;

    extern __shared__ half smemPtr[];

    SmemTensor4D<typeA, stages, tileSize.x / 8, tileSize.z / 8, 32> smemA(smemPtr); // each thread accesses contiguous 32 bytes, 4 x bank conflict maybe?
    SmemTensor3D<typeB, stages, converterB.physical_shape().x, converterB.physical_shape().y> smemB(smemA.endPtr);
    SmemTensor3D<half, warpCount.x * warpCount.y, 8, 32 + paddingC> smemC(smemB.endPtr);
    SmemTensor1D<typeCode, codeSize> smemCode(smemC.endPtr);
    SmemTensor1D<typeAbsMax, absMaxPerBlock> smemAbsMax(smemCode.endPtr);

    load_abs_max<typeAbsMax, absMaxPerBlock, threadsPerBlock>(smemAbsMax, absMaxPtr, globalStart.x, tid);
    load_code_v2<typeCode, codeSize, threadsPerBlock>(smemCode, codePtr, tid);
    __syncthreads();

    Data4B accumulator[warpMmaCount.x][warpMmaCount.y][accBytes];
    Data4B fill_value;
    fill_value.as_half2[0] = __float2half2_rn(0.f);
    fill_accumulator<
        warpMmaCount.x,
        warpMmaCount.y,
        accBytes>(accumulator, fill_value.as_int32[0]);

    Deq4GemmIterator<
        tileSize.x, tileSize.y, tileSize.z,
        transformA, transformB,
        quantStatA, quantStatB,
        typeA, typeB,
        alignSizeA, alignSizeB,
        contiguousBytesA, contiguousBytesB,
        warpCount.x, warpCount.y,
        mmaSize.x, mmaSize.y, mmaSize.z,
        warpMmaCount.x, warpMmaCount.y, warpMmaCount.z,
        stages,
        absMaxPerBlock,
        codeSize,
        converterA.physical_shape().x, converterA.physical_shape().y,
        converterB.physical_shape().x, converterB.physical_shape().y,
        isSafe,
        pipelineStrat,
        accMode>
        gemmIterator(matA, matB, problemSize, globalStart, globalEnd, leadingDims);

    int useless = 0;
    gemmIterator.run(smemA, smemB, smemCode, smemAbsMax, accumulator, useless);
    if (useless > 0)
        useless = 0;
    accumulator[0][0][0].as_half[0] += half(useless);

    store_F<
        mmaSize.x,
        mmaSize.y,
        mmaSize.z,
        warpCount.x,
        warpCount.y,
        warpMmaCount.x,
        warpMmaCount.y,
        warpMmaCount.z,
        paddingC,
        accMode,
        splitK>(
        accumulator,
        matC,
        smemC,
        {globalStart.y, globalStart.x},
        {problemSize.y, problemSize.x},
        {leadingDims.z});
}

std::tuple<int, int, int> calculateGridDims(int n, int m, int tile_n, int tile_m, int patch_n, int patch_m, int split_k)
{

    int n_ = div_ru(n, tile_n * patch_n);
    int m_ = div_ru(m, tile_m * patch_m);

    return std::make_tuple(patch_n * patch_m, n_ * m_, split_k);
}

constexpr unsigned int requiredSmem(const KernelConfig config)
{

    unsigned int res = 0;
    res += config.tileM * config.tileK * config.stages / 2;
    res += config.tileN * config.tileK * config.stages * 2;
    res += config.warpCountM * config.warpCountN * 8 * (32 + config.paddingC) * 2;
    res += config.absMaxPerBlock * 2;
    res += config.codeSize * 2;
    return res;
}

#define LAUNCH_KERNEL_IF_CONDITION(config, mCond, nMinCond, nMaxCond, kCond)                                                  \
    else if (m == mCond && n >= nMinCond && n <= nMaxCond && k == kCond)                                                      \
    {                                                                                                                         \
        auto kernelFunc = Tuna<                                                                                               \
            config.alignSizeBytesA, config.alignSizeBytesB, config.tileM, config.tileN, config.tileK, config.patchM,          \
            config.patchN, config.k, config.m, config.splitK, config.warpCountM, config.warpCountN, config.mmaSizeM,          \
            config.mmaSizeN, config.mmaSizeK, config.warpMmaCountM, config.warpMmaCountN, config.warpMmaCountK,               \
            config.contiguousBytesA, config.contiguousBytesB, config.deqBlockSize, config.stages, config.absMaxPerBlock,      \
            config.threadsPerBlock, config.codeSize, config.isSafe, config.pipelineStrat, config.paddingC, config.transformA, \
            config.transformB, config.transformC, config.quantStatA, config.quantStatB>;                                      \
                                                                                                                              \
        constexpr unsigned int smem = requiredSmem(config);                                                                   \
                                                                                                                              \
        cudaFuncSetAttribute(                                                                                                 \
            kernelFunc,                                                                                                       \
            cudaFuncAttributeMaxDynamicSharedMemorySize,                                                                      \
            smem);                                                                                                            \
                                                                                                                              \
        constexpr int m_ = div_ru(mCond, config.tileM * config.patchM);                                                       \
        const int n_ = div_ru(n, config.tileN * config.patchN);                                                               \
        dim3 blocks_per_grid(config.patchN *config.patchM, n_ *m_, config.splitK);                                            \
        constexpr dim3 threads_per_block(config.threadsPerBlock);                                                             \
        kernelFunc<<<blocks_per_grid, threads_per_block, smem>>>(Q_ptr, X_ptr, C_ptr, S_ptr, O_ptr, n);                       \
        return;                                                                                                               \
    }

void wrapper(void *q, void *x, void *c, void *s, void *o, const int m, const int n, const int k)
{

    const uint8_t *Q_ptr = reinterpret_cast<const uint8_t *>(q);
    const half *X_ptr = reinterpret_cast<const half *>(x);
    const float *C_ptr = reinterpret_cast<const float *>(c);
    const float *S_ptr = reinterpret_cast<const float *>(s);
    half *O_ptr = reinterpret_cast<half *>(o);

    if (false)
    {
    }

    LAUNCH_KERNEL_IF_CONDITION(attn_out_1, 4096, 0, 15, 4096)
    LAUNCH_KERNEL_IF_CONDITION(qkv_proj_1, 6144, 0, 15, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_in_1, 28672, 0, 15, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_in_1, 229376, 0, 15, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_out_1, 4096, 0, 15, 14336)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_out_1, 32768, 0, 15, 14336)

    LAUNCH_KERNEL_IF_CONDITION(attn_out_16, 4096, 16, 31, 4096)
    LAUNCH_KERNEL_IF_CONDITION(qkv_proj_16, 6144, 16, 31, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_in_16, 28672, 16, 31, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_in_16, 229376, 16, 31, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_out_16, 4096, 16, 31, 14336)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_out_16, 32768, 16, 31, 14336)

    LAUNCH_KERNEL_IF_CONDITION(attn_out_32, 4096, 32, 63, 4096)
    LAUNCH_KERNEL_IF_CONDITION(qkv_proj_32, 6144, 32, 63, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_in_32, 28672, 32, 63, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_in_32, 229376, 32, 63, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_out_32, 4096, 32, 63, 14336)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_out_32, 32768, 32, 63, 14336)

    LAUNCH_KERNEL_IF_CONDITION(attn_out_64, 4096, 64, 127, 4096)
    LAUNCH_KERNEL_IF_CONDITION(qkv_proj_64, 6144, 64, 127, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_in_64, 28672, 64, 127, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_in_64, 229376, 64, 127, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_out_64, 4096, 64, 127, 14336)

    LAUNCH_KERNEL_IF_CONDITION(attn_out_128, 4096, 128, 255, 4096)
    LAUNCH_KERNEL_IF_CONDITION(qkv_proj_128, 6144, 128, 255, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_in_128, 28672, 128, 255, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_in_128, 229376, 128, 255, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_out_128, 4096, 128, 255, 14336)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_out_128, 32768, 128, 255, 14336)

    LAUNCH_KERNEL_IF_CONDITION(attn_out_256, 4096, 256, 128000, 4096)
    LAUNCH_KERNEL_IF_CONDITION(qkv_proj_256, 6144, 256, 128000, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_in_256, 28672, 256, 128000, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_in_256, 229376, 256, 128000, 4096)
    LAUNCH_KERNEL_IF_CONDITION(mistral_mlp_out_256, 4096, 256, 128000, 14336)
    LAUNCH_KERNEL_IF_CONDITION(mixtral_mlp_out_256, 32768, 256, 128000, 14336)
}
