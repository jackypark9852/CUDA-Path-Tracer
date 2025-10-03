/*

 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2025  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include "NtcTextureSet.h"
#include "NtcInference.h"


static __forceinline__ __device__ int CLAMP( int x, int a, int b )
{
    return ( x <= a ) ? a : ( x >= b ) ? b : x;
}

template <class T_VEC_OUT>
__device__ __forceinline__ 
static bool inferTexel( T_VEC_OUT& out, NtcTextureSetConstants& tsc, uint8_t* latents, uint8_t* mlpWeights, int x, int y, int lod )
{
    if (params.bound.enableFP8) {
        switch (params.bound.networkVersion) {
            case NTC_NETWORK_SMALL:  return inferTexelCoopVec_fp8<T_VEC_OUT, NTC_NETWORK_SMALL> ( out, tsc, latents, mlpWeights, x, y, lod ); break;
            case NTC_NETWORK_MEDIUM: return inferTexelCoopVec_fp8<T_VEC_OUT, NTC_NETWORK_MEDIUM>( out, tsc, latents, mlpWeights, x, y, lod ); break;
            case NTC_NETWORK_LARGE:  return inferTexelCoopVec_fp8<T_VEC_OUT, NTC_NETWORK_LARGE> ( out, tsc, latents, mlpWeights, x, y, lod ); break;
            case NTC_NETWORK_XLARGE: return inferTexelCoopVec_fp8<T_VEC_OUT, NTC_NETWORK_XLARGE>( out, tsc, latents, mlpWeights, x, y, lod ); break;
        }
    } else {
        switch (params.bound.networkVersion) {
            case NTC_NETWORK_SMALL:  return inferTexelCoopVec_int8<T_VEC_OUT, NTC_NETWORK_SMALL> ( out, tsc, latents, mlpWeights, x, y, lod ); break;
            case NTC_NETWORK_MEDIUM: return inferTexelCoopVec_int8<T_VEC_OUT, NTC_NETWORK_MEDIUM>( out, tsc, latents, mlpWeights, x, y, lod ); break;
            case NTC_NETWORK_LARGE:  return inferTexelCoopVec_int8<T_VEC_OUT, NTC_NETWORK_LARGE> ( out, tsc, latents, mlpWeights, x, y, lod ); break;
            case NTC_NETWORK_XLARGE: return inferTexelCoopVec_int8<T_VEC_OUT, NTC_NETWORK_XLARGE>( out, tsc, latents, mlpWeights, x, y, lod ); break;
        }
    }
    return false;
}

template <class T_VEC_OUT>
__forceinline__ __device__ 
static bool ntcTex2D( T_VEC_OUT& out, NtcTextureSet* ntsPtr, float u, float v, float2 xi )
{
    const float2 texelJitter = xi - float2{ 0.5f, 0.5f };  // stochastic bilinear sampling

    const NtcTextureSet& nts = *ntsPtr;

    const int i = CLAMP( u * nts.constants.imageWidth + texelJitter.x, 0, nts.constants.imageWidth - 1 );
    const int j = CLAMP( v * nts.constants.imageHeight + texelJitter.y, 0, nts.constants.imageHeight - 1 );

    const int mipLevel = 0;

    return inferTexel( out, ntsPtr->constants, nts.d_latents, nts.d_mlpWeights, i, j, mipLevel );
}
