/*
 * whirlcoin kernel implementation.
 *
 * ==========================(LICENSE BEGIN)============================
 *
 * Copyright (c) 2014  phm
 * Copyright (c) 2014 djm34
  * Copyright (c) 2014 uray
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ===========================(LICENSE END)=============================
 *
 * @author   djm34, uray
 */
#ifndef W_CL
#define W_CL

#if __ENDIAN_LITTLE__
#define SPH_LITTLE_ENDIAN 1
#else
#define SPH_BIG_ENDIAN 1
#endif

#define SPH_UPTR sph_u64

typedef unsigned int sph_u32;
typedef int sph_s32;
#ifndef __OPENCL_VERSION__
typedef unsigned long long sph_u64 __attribute__ ((aligned (128)));
typedef long long sph_s64;
#else
typedef unsigned long sph_u64;
typedef long sph_s64;
#endif

#define SPH_64 1
#define SPH_64_TRUE 1

#define SPH_C32(x)    ((sph_u32)(x ## U))
#define SPH_C64(x)    ((sph_u64)(x ## UL))

#define SPH_T32(x) (as_uint(x))
#define SPH_ROTL32(x, n) rotate(as_uint(x), as_uint(n))
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))
#define SPH_T64(x) (as_ulong(x))
#define SPH_ROTL64(x, n) rotate(as_ulong(x), (n) & 0xFFFFFFFFFFFFFFFFUL)
#define SPH_ROTR64(x, n)   SPH_ROTL64(x, (64 - (n)))

#include "whirlpool.cl"

#define SWAP4(x) (SPH_ROTL32(as_uint(x) & 0x00FF00FF, 24U)|SPH_ROTL32(as_uint(x) & 0xFF00FF00, 8U))
#define SWAP8(x) as_ulong(as_uchar8(x).s76543210)

#if SPH_BIG_ENDIAN
    #define DEC64E(x) (x)
    #define DEC64BE(x) (*(const __global sph_u64 *) (x));
    #define DEC32LE(x) SWAP4(*(const __global sph_u32 *) (x));
    #define DEC64LE(x) SWAP8(*(const __global sph_u64 *) (x));
#else
    #define DEC64E(x) SWAP8(x)
    #define DEC64BE(x) SWAP8(*(const __global sph_u64 *) (x));
    #define DEC32LE(x) (*(const __global sph_u32 *) (x));
    #define DEC64LE(x) (*(const __global sph_u64 *) (x));
#endif

typedef union {
    unsigned char h1[64];
    uint h4[16];
    ulong h8[8];
} hash_t;

void whirlpool_round(sph_u64* n, sph_u64* h){
    sph_u64 t0, t1, t2, t3, t4, t5, t6, t7;
    for (unsigned r = 0; r < 10; r ++) {
        t0 = (ROUND_ELT(h, 0, 7, 6, 5, 4, 3, 2, 1) ^ rc[r]); 
        t1 = (ROUND_ELT(h, 1, 0, 7, 6, 5, 4, 3, 2) ^ 0 ); 
        t2 = (ROUND_ELT(h, 2, 1, 0, 7, 6, 5, 4, 3) ^ 0 ); 
        t3 = (ROUND_ELT(h, 3, 2, 1, 0, 7, 6, 5, 4) ^ 0 ); 
        t4 = (ROUND_ELT(h, 4, 3, 2, 1, 0, 7, 6, 5) ^ 0 );  
        t5 = (ROUND_ELT(h, 5, 4, 3, 2, 1, 0, 7, 6) ^ 0 );  
        t6 = (ROUND_ELT(h, 6, 5, 4, 3, 2, 1, 0, 7) ^ 0 );  
        t7 = (ROUND_ELT(h, 7, 6, 5, 4, 3, 2, 1, 0) ^ 0 );  

        h[0] = t0;
        h[1] = t1;
        h[2] = t2;
        h[3] = t3;
        h[4] = t4;
        h[5] = t5;
        h[6] = t6;
        h[7] = t7;

        t0 = ROUND_ELT(n, 0, 7, 6, 5, 4, 3, 2, 1) ^ h[0]; 
        t1 = ROUND_ELT(n, 1, 0, 7, 6, 5, 4, 3, 2) ^ h[1]; 
        t2 = ROUND_ELT(n, 2, 1, 0, 7, 6, 5, 4, 3) ^ h[2]; 
        t3 = ROUND_ELT(n, 3, 2, 1, 0, 7, 6, 5, 4) ^ h[3]; 
        t4 = ROUND_ELT(n, 4, 3, 2, 1, 0, 7, 6, 5) ^ h[4]; 
        t5 = ROUND_ELT(n, 5, 4, 3, 2, 1, 0, 7, 6) ^ h[5]; 
        t6 = ROUND_ELT(n, 6, 5, 4, 3, 2, 1, 0, 7) ^ h[6]; 
        t7 = ROUND_ELT(n, 7, 6, 5, 4, 3, 2, 1, 0) ^ h[7]; 

        n[0] = t0;
        n[1] = t1;
        n[2] = t2;
        n[3] = t3;
        n[4] = t4;
        n[5] = t5;
        n[6] = t6;
        n[7] = t7;    
    }
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(__global unsigned char* block, __global hash_t* hashes)
{
   uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);
  

    sph_u64 n[8]; 
    sph_u64 h[8];
    sph_u64 state[8];

    h[0] = h[1] = h[2] = h[3] = h[4] = h[5] = h[6] = h[7] = 0;

    n[0] =  h[0] ^ DEC64LE(block +    0);
    n[1] =  h[1] ^ DEC64LE(block +    8);
    n[2] =  h[2] ^ DEC64LE(block +   16);
    n[3] =  h[3] ^ DEC64LE(block +   24);
    n[4] =  h[4] ^ DEC64LE(block +   32);
    n[5] =  h[5] ^ DEC64LE(block +   40);
    n[6] =  h[6] ^ DEC64LE(block +   48);
    n[7] =  h[7] ^ DEC64LE(block +   56);

    whirlpool_round(n, h);

    h[0] = state[0] = n[0] ^ DEC64LE(block +   0);
    h[1] = state[1] = n[1] ^ DEC64LE(block +   8);
    h[2] = state[2] = n[2] ^ DEC64LE(block +   16);
    h[3] = state[3] = n[3] ^ DEC64LE(block +   24);
    h[4] = state[4] = n[4] ^ DEC64LE(block +   32);
    h[5] = state[5] = n[5] ^ DEC64LE(block +   40);
    h[6] = state[6] = n[6] ^ DEC64LE(block +   48);
    h[7] = state[7] = n[7] ^ DEC64LE(block +   56);

    
    n[0] = DEC64LE(block +  64);
    n[1] = DEC64LE(block +  72);
    n[1] &= 0x00000000FFFFFFFF;
    n[1] ^= ((sph_u64) gid) << 32;
    n[3] = n[4] = n[5] = n[6] = 0;
    n[2] = 0x0000000000000080; 
    n[7] = 0x8002000000000000;
    sph_u64 temp0,temp1,temp2,temp7;
    temp0 = n[0];
    temp1 = n[1];
    temp2 = n[2];
    temp7 = n[7];

    n[0] ^= h[0];
    n[1] ^= h[1];
    n[2] ^= h[2];
    n[3] ^= h[3];
    n[4] ^= h[4];
    n[5] ^= h[5];
    n[6] ^= h[6];
    n[7] ^= h[7];

    whirlpool_round(n, h);
    
    hash->h8[0] = state[0] ^ n[0] ^ temp0;
    hash->h8[1] = state[1] ^ n[1] ^ temp1;
    hash->h8[2] = state[2] ^ n[2] ^ temp2;
    hash->h8[3] = state[3] ^ n[3];
    hash->h8[4] = state[4] ^ n[4];
    hash->h8[5] = state[5] ^ n[5];
    hash->h8[6] = state[6] ^ n[6];
    hash->h8[7] = state[7] ^ n[7] ^ temp7;
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search1(__global hash_t* hashes)
{
   uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);

    sph_u64 n[8]; 
    sph_u64 h[8];
    sph_u64 state[8];

    for (int loop=0;loop<3; loop++) {
        n[0] = (hash->h8[0]);
        n[1] = (hash->h8[1]);
        n[2] = (hash->h8[2]);
        n[3] = (hash->h8[3]);
        n[4] = (hash->h8[4]);
        n[5] = (hash->h8[5]);
        n[6] = (hash->h8[6]);
        n[7] = (hash->h8[7]);

        h[0] = h[1] = h[2] = h[3] = h[4] = h[5] = h[6] = h[7] = 0;

        n[0] ^= h[0];
        n[1] ^= h[1];
        n[2] ^= h[2];
        n[3] ^= h[3];
        n[4] ^= h[4];
        n[5] ^= h[5];
        n[6] ^= h[6];
        n[7] ^= h[7];

        whirlpool_round(n, h);

        n[0] = h[0] = state[0] = n[0] ^ (hash->h8[0]);
        n[1] = h[1] = state[1] = n[1] ^ (hash->h8[1]);
        n[2] = h[2] = state[2] = n[2] ^ (hash->h8[2]);
        n[3] = h[3] = state[3] = n[3] ^ (hash->h8[3]);
        n[4] = h[4] = state[4] = n[4] ^ (hash->h8[4]);
        n[5] = h[5] = state[5] = n[5] ^ (hash->h8[5]);
        n[6] = h[6] = state[6] = n[6] ^ (hash->h8[6]);
        n[7] = h[7] = state[7] = n[7] ^ (hash->h8[7]);

        n[0] ^= 0x80 ;
        n[1] ^= 0 ;
        n[2] ^= 0 ;
        n[3] ^= 0 ;
        n[4] ^= 0 ;
        n[5] ^= 0 ;
        n[6] ^= 0 ;
        n[7] ^= 0x2000000000000 ;

        whirlpool_round(n, h);

        hash->h8[0] = state[0] ^ n[0] ^ 0x80;
        hash->h8[1] = state[1] ^ n[1];
        hash->h8[2] = state[2] ^ n[2];
        hash->h8[3] = state[3] ^ n[3];
        hash->h8[4] = state[4] ^ n[4];
        hash->h8[5] = state[5] ^ n[5];
        hash->h8[6] = state[6] ^ n[6];
        hash->h8[7] = state[7] ^ n[7] ^ 0x2000000000000;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);
}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search2(__global hash_t* hashes)
{


}

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search3(__global hash_t* hashes, __global uint* output, const ulong target)
{
    uint gid = get_global_id(0);
    __global hash_t *hash = &(hashes[gid-get_global_offset(0)]);

    bool result = (hash->h8[3] <= target);
    if (result)
        output[atomic_inc(output+0xFF)] = SWAP4(gid);
}

#endif // W_CL