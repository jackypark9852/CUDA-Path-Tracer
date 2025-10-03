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

#include <cuda/helpers.h>
#include <optix.h>
#include <optix_device.h>

#include <libntc/shaders/DecompressConstants.h>


constexpr OptixCoopVecMatrixLayout MAT_LAYOUT = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;

typedef struct
{
    half x, y, z, w;
} half4;

template <typename VecT>
__device__ __forceinline__ half4  make_half4 ( VecT v  ) { return { v.x, v.y, v.z, v.w }; }
__device__ __forceinline__ half4  make_half4 ( float x ) { return { half(x), half(x), half(x), half(x) }; }
__device__ __forceinline__ half4  make_half4 ( int x   ) { return { half(x), half(x), half(x), half(x) }; }
__device__ __forceinline__ half4  make_half4 ( float a, float b, float c, float d ) { return { half(a), half(b), half(c), half(d) }; }
__device__ __forceinline__ float4 make_float4( half4 v ) { return { float(v.x), float(v.y), float(v.z), float(v.w) }; }

__device__ __forceinline__ half4 operator+( const half4& a, const half4& b ) { return { a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w }; }
__device__ __forceinline__ half4 operator*( const half4& a, const half4& b ) { return { a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w }; }
__device__ __forceinline__ half4 operator*( const half4& a, const half b   ) { return { a.x*b,   a.y*b,   a.z*b,   a.w*b   }; }


template <int NETWORK_VERSION>
struct NTC_PARAMS
{
    static const int INPUT_CHANNELS = ( NETWORK_VERSION == NTC_NETWORK_SMALL )  ? NTC_MLP_INPUT_CHANNELS_SMALL :
                                      ( NETWORK_VERSION == NTC_NETWORK_MEDIUM ) ? NTC_MLP_INPUT_CHANNELS_MEDIUM :
                                      ( NETWORK_VERSION == NTC_NETWORK_LARGE )  ? NTC_MLP_INPUT_CHANNELS_LARGE :
                                      ( NETWORK_VERSION == NTC_NETWORK_XLARGE ) ? NTC_MLP_INPUT_CHANNELS_XLARGE :
                                                                                  0;  // Unsupported value

    static const int HR_FEATURES = ( NETWORK_VERSION == NTC_NETWORK_SMALL )  ? NTC_MLP_HR_FEATURES_SMALL :
                                   ( NETWORK_VERSION == NTC_NETWORK_MEDIUM ) ? NTC_MLP_HR_FEATURES_MEDIUM :
                                   ( NETWORK_VERSION == NTC_NETWORK_LARGE )  ? NTC_MLP_HR_FEATURES_LARGE :
                                   ( NETWORK_VERSION == NTC_NETWORK_XLARGE ) ? NTC_MLP_HR_FEATURES_XLARGE :
                                                                               0;  // Unsupported value

    static const int LR_FEATURES = NTC_MLP_LR_FEATURES;

    static const int SAMPLED_FEATURES_HR    = HR_FEATURES * 4;
    static const int SAMPLED_FEATURES_LR    = LR_FEATURES;
    static const int SAMPLED_FEATURES_TOTAL = SAMPLED_FEATURES_HR + SAMPLED_FEATURES_LR;

    static const int HIDDEN_LAYER_CHANNELS = NTC_MLP_HIDDEN_CHANNELS;

    static const int OUTPUT_CHANNELS = NTC_MLP_OUTPUT_CHANNELS;

    static const int LATENTS_COUNT     = HR_FEATURES + LR_FEATURES;
    static const int HR_LATENTS_WIDTH  = DECOMPRESS_CS_BLOCK_WIDTH + 2;
    static const int HR_LATENTS_HEIGHT = DECOMPRESS_CS_BLOCK_HEIGHT + 2;
    static const int LR_LATENTS_WIDTH  = DECOMPRESS_CS_BLOCK_WIDTH / 2 + 2;
    static const int LR_LATENTS_HEIGHT = DECOMPRESS_CS_BLOCK_HEIGHT / 2 + 2;
    static const int MAX_INPUT_SIZE = ( INPUT_CHANNELS > HIDDEN_LAYER_CHANNELS ) ? INPUT_CHANNELS : HIDDEN_LAYER_CHANNELS;
    static const int MAX_OUTPUT_SIZE = ( HIDDEN_LAYER_CHANNELS > OUTPUT_CHANNELS ) ? HIDDEN_LAYER_CHANNELS : OUTPUT_CHANNELS;
};

inline __device__
float frac( float x )
{
    return x - floorf( x );
}


/*
inline __device__
bool IsHalfSpecial( half f )
{
    uint16_t u = __half_as_ushort( f );
    // Test if the number is an IEEE 754 Inf or NaN pattern (all 1's exponent)
    return ( u & 0x7c00 ) == 0x7c00;
}
*/


template<typename T>
using PackedType = typename
std::conditional<
    std::is_same<T, half>::value, half2,
    std::conditional<
    std::is_same<T, char>::value, char4,
    std::conditional<
    std::is_same<T, unsigned char>::value, uchar4, T
    >
    >
>::type;

template <typename T>
__device__ inline
PackedType<T> unpack( uint32_t x )
{
    return *( (PackedType<T>*)( &x ) );
}

template <typename T>
__device__ inline
uint32_t pack( PackedType<T> x )
{
    return *( (uint32_t*)( &x ) );
}


__device__ __forceinline__
int2 GetTextureDimensions( NtcTextureSetConstants& tsc, int mipLevel )
{
    return make_int2( max( tsc.imageWidth >> mipLevel, 1 ), max( tsc.imageHeight >> mipLevel, 1 ) );
}


__device__ __forceinline__
uint4 LoadFourRawLatents(
    const uint8_t*                    buffer,
    uint32_t                          bufferOffset,
    const NtcLatentEncodingConstants& encoding,
    const NtcNeuralMipConstants&      neuralMip,
    int                               addr )
{
    uint32_t index = bufferOffset + neuralMip.dataOffset + ( addr >> encoding.logElementsPerUint ) * 4;
    uint32_t word  = *(uint32_t*)( buffer + index );

    const uint32_t firstOffset = ( addr & encoding.addressMask ) * encoding.quantBits;

    word = word >> firstOffset;
    const uint32_t bits0 = word & encoding.dataMask; word = word >> encoding.quantBits;
    const uint32_t bits1 = word & encoding.dataMask; word = word >> encoding.quantBits;
    const uint32_t bits2 = word & encoding.dataMask; word = word >> encoding.quantBits;
    const uint32_t bits3 = word & encoding.dataMask;

    return make_uint4( bits0, bits1, bits2, bits3 );
}


__device__ __forceinline__
int4 LoadFourInputQuantizedLatents(
    const uint8_t*                    buffer,
    uint32_t                          bufferOffset,
    const NtcLatentEncodingConstants& encoding,
    const NtcNeuralMipConstants&      neuralMip,
    int                               addr )
{
    uint4 bits = LoadFourRawLatents( buffer, bufferOffset, encoding, neuralMip, addr );
    return make_int4( bits.x, bits.y, bits.z, bits.w ) * make_int4( encoding.quantizedScale ) + make_int4( encoding.quantizedBias );
}


__device__ __forceinline__
half4 LoadFourInputQuantizedLatents_FP16(
    const uint8_t*                    buffer,
    uint32_t                          bufferOffset,
    const NtcLatentEncodingConstants& encoding,
    const NtcNeuralMipConstants&      neuralMip,
    int                               addr )
{
    uint4 bits  = LoadFourRawLatents( buffer, bufferOffset, encoding, neuralMip, addr );
    half4 hbits = { half( bits.x ), half( bits.y ), half( bits.z ), half( bits.w ) };
    return make_half4( bits ) * make_half4( __int_as_float( encoding.quantizedScale ) )
           + make_half4( __int_as_float( encoding.quantizedBias ) );
}


__device__ __forceinline__
float4 SetupLatentBilinearFilter(
    const NtcNeuralMipConstants& neuralMip,
    float2 uv,
    int2& outTopLeftPos)
{
    float2 pixelPos = uv * make_float2( neuralMip.imageWidth, neuralMip.imageHeight ) - 0.5f
                      - make_float2( neuralMip.sliceLeft, neuralMip.sliceTop );

    float2 topLeftPos = floor( pixelPos );

    const float2 d  = pixelPos - topLeftPos;
    const float2 dn = 1 - d;

    outTopLeftPos = make_int2( topLeftPos );

    return make_float4( dn.x * dn.y, d.x * dn.y, dn.x * d.y, d.x * d.y );
}


__device__ __forceinline__
int4 operator>>( const int4& x, int n )
{
    return make_int4( x.x >> n, x.y >> n, x.z >> n, x.w >> n );
}


template<int NUM_FEATURES, bool ALL_CORNERS, class vecT>
__device__ __forceinline__
bool SampleLatentGrid(
    const uint8_t* __restrict         latentsBuffer,
    uint32_t                          bufferOffset,
    const NtcLatentEncodingConstants& encoding,
    const NtcNeuralMipConstants&      neuralMip,
    float2                            uv,
    int                               outputOffset,
    vecT&                             outputArray )
{
    int2   topLeftPos;
    float4 weights  = SetupLatentBilinearFilter( neuralMip, uv, topLeftPos );
    int4   iweights = make_int4( weights * 256.f );

    // Shift right the interpolated weights by 8 to undo the 256 factor above
    const int normalizationShift = 8;

    const int x0 = min( max( topLeftPos.x, 0 ), neuralMip.sliceWidth - 1 );
    const int y0 = min( max( topLeftPos.y, 0 ), neuralMip.sliceHeight - 1 );
    const int x1 = min( max( topLeftPos.x + 1, 0 ), neuralMip.sliceWidth - 1 );
    const int y1 = min( max( topLeftPos.y + 1, 0 ), neuralMip.sliceHeight - 1 );

    int a00 = (y0 * neuralMip.sliceWidth + x0) * encoding.numFeatures;
    int a01 = (y0 * neuralMip.sliceWidth + x1) * encoding.numFeatures;
    int a10 = (y1 * neuralMip.sliceWidth + x0) * encoding.numFeatures;
    int a11 = (y1 * neuralMip.sliceWidth + x1) * encoding.numFeatures;

#pragma unroll
    for (int i = 0; i < NUM_FEATURES; i += 4)
    {
        if (i >= encoding.numFeatures) break;

        const int4 x00 = LoadFourInputQuantizedLatents(latentsBuffer, bufferOffset, encoding, neuralMip, a00) * iweights.x;
        const int4 x01 = LoadFourInputQuantizedLatents(latentsBuffer, bufferOffset, encoding, neuralMip, a01) * iweights.y;
        const int4 x10 = LoadFourInputQuantizedLatents(latentsBuffer, bufferOffset, encoding, neuralMip, a10) * iweights.z;
        const int4 x11 = LoadFourInputQuantizedLatents(latentsBuffer, bufferOffset, encoding, neuralMip, a11) * iweights.w;

        a00 += 4; a01 += 4; a10 += 4; a11 += 4;

        if (ALL_CORNERS)
        {
            // Copy the latents for the 4 pixels into the network inputs.
            outputArray[outputOffset + NUM_FEATURES * 0 + i + 0] = x00.x >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 0 + i + 1] = x00.y >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 0 + i + 2] = x00.z >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 0 + i + 3] = x00.w >> normalizationShift;

            outputArray[outputOffset + NUM_FEATURES * 1 + i + 0] = x01.x >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 1 + i + 1] = x01.y >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 1 + i + 2] = x01.z >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 1 + i + 3] = x01.w >> normalizationShift;

            outputArray[outputOffset + NUM_FEATURES * 2 + i + 0] = x10.x >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 2 + i + 1] = x10.y >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 2 + i + 2] = x10.z >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 2 + i + 3] = x10.w >> normalizationShift;

            outputArray[outputOffset + NUM_FEATURES * 3 + i + 0] = x11.x >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 3 + i + 1] = x11.y >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 3 + i + 2] = x11.z >> normalizationShift;
            outputArray[outputOffset + NUM_FEATURES * 3 + i + 3] = x11.w >> normalizationShift;
        }
        else
        {
            // Blend the features of the 4 pixels.
            int4 d = (x00 + x01 + x10 + x11) >> normalizationShift;
            outputArray[outputOffset + i + 0] = d.x;
            outputArray[outputOffset + i + 1] = d.y;
            outputArray[outputOffset + i + 2] = d.z;
            outputArray[outputOffset + i + 3] = d.w;
        }
    }

    return true;
}


template<int NUM_FEATURES, bool ALL_CORNERS, class vecT>
__device__ __forceinline__
bool SampleLatentGrid_FP16(
    const uint8_t* __restrict         latentsBuffer,
    uint32_t                          bufferOffset,
    const NtcLatentEncodingConstants& encoding,
    const NtcNeuralMipConstants&      neuralMip,
    float2                            uv,
    int                               outputOffset,
    vecT&                             outputArray )
{
    int2   topLeftPos;
    float4 weights  = SetupLatentBilinearFilter( neuralMip, uv, topLeftPos );
    half4  hweights = { half( weights.x ), half( weights.y ), half( weights.z ), half( weights.w ) };

    const int x0 = min( max( topLeftPos.x, 0 ), neuralMip.sliceWidth - 1 );
    const int y0 = min( max( topLeftPos.y, 0 ), neuralMip.sliceHeight - 1 );
    const int x1 = min( max( topLeftPos.x + 1, 0 ), neuralMip.sliceWidth - 1 );
    const int y1 = min( max( topLeftPos.y + 1, 0 ), neuralMip.sliceHeight - 1 );

    int a00 = ( y0 * neuralMip.sliceWidth + x0 ) * encoding.numFeatures;
    int a01 = ( y0 * neuralMip.sliceWidth + x1 ) * encoding.numFeatures;
    int a10 = ( y1 * neuralMip.sliceWidth + x0 ) * encoding.numFeatures;
    int a11 = ( y1 * neuralMip.sliceWidth + x1 ) * encoding.numFeatures;

#pragma unroll
    for (int i = 0; i < NUM_FEATURES; i += 4)
    {
        if (i >= encoding.numFeatures) break;

        half4 x00 = LoadFourInputQuantizedLatents_FP16(latentsBuffer, bufferOffset, encoding, neuralMip, a00) * hweights.x; a00 += 4;
        half4 x01 = LoadFourInputQuantizedLatents_FP16(latentsBuffer, bufferOffset, encoding, neuralMip, a01) * hweights.y; a01 += 4;
        half4 x10 = LoadFourInputQuantizedLatents_FP16(latentsBuffer, bufferOffset, encoding, neuralMip, a10) * hweights.z; a10 += 4;
        half4 x11 = LoadFourInputQuantizedLatents_FP16(latentsBuffer, bufferOffset, encoding, neuralMip, a11) * hweights.w; a11 += 4;

        if (ALL_CORNERS)
        {
            // Copy the latents for the 4 pixels into the network inputs.
            outputArray[outputOffset + NUM_FEATURES * 0 + i + 0] = x00.x;
            outputArray[outputOffset + NUM_FEATURES * 0 + i + 1] = x00.y;
            outputArray[outputOffset + NUM_FEATURES * 0 + i + 2] = x00.z;
            outputArray[outputOffset + NUM_FEATURES * 0 + i + 3] = x00.w;

            outputArray[outputOffset + NUM_FEATURES * 1 + i + 0] = x01.x;
            outputArray[outputOffset + NUM_FEATURES * 1 + i + 1] = x01.y;
            outputArray[outputOffset + NUM_FEATURES * 1 + i + 2] = x01.z;
            outputArray[outputOffset + NUM_FEATURES * 1 + i + 3] = x01.w;

            outputArray[outputOffset + NUM_FEATURES * 2 + i + 0] = x10.x;
            outputArray[outputOffset + NUM_FEATURES * 2 + i + 1] = x10.y;
            outputArray[outputOffset + NUM_FEATURES * 2 + i + 2] = x10.z;
            outputArray[outputOffset + NUM_FEATURES * 2 + i + 3] = x10.w;

            outputArray[outputOffset + NUM_FEATURES * 3 + i + 0] = x11.x;
            outputArray[outputOffset + NUM_FEATURES * 3 + i + 1] = x11.y;
            outputArray[outputOffset + NUM_FEATURES * 3 + i + 2] = x11.z;
            outputArray[outputOffset + NUM_FEATURES * 3 + i + 3] = x11.w;
        }
        else
        {
            // Blend the features of the 4 pixels.
            half4 d = (x00 + x01 + x10 + x11);
            outputArray[outputOffset + i + 0] = d.x;
            outputArray[outputOffset + i + 1] = d.y;
            outputArray[outputOffset + i + 2] = d.z;
            outputArray[outputOffset + i + 3] = d.w;
        }
    }

    return true;
}


__device__ __forceinline__
float4 EvaluatePositionalEncoding( float2 posf, float iscale )
{
    float4 result;

    result.x = frac( posf.x * iscale ) * 4;
    result.x = abs( result.x - 2 ) - 1;
    result.y = frac( posf.y * iscale ) * 4;
    result.y = abs( result.y - 2 ) - 1;

    result.z = frac( posf.x * iscale + 0.25f ) * 4;
    result.z = abs( result.z - 2 ) - 1;
    result.w = frac( posf.y * iscale + 0.25f ) * 4;
    result.w = abs( result.w - 2 ) - 1;

    return result;
}

__device__ __forceinline__
uint32_t FloatToInt8(float h, float scale)
{
    return uint32_t(int(clamp(h * scale, -128.f, 127.f)) & 0xff);
}


template<class vecT>
__device__ __forceinline__
void EncodeSamplePosition(float2 posf, float lod, int offset, vecT& outputArray)
{
    int idx = offset;
    int scale = NTC_MLP_POS_ENC_SCALE;
    float iscale = 1.f / scale;

    static const float c_InputScale = 127.5f; // Inputs are in the [-1, 1] range, scale matches tin::InputQuant

#pragma unroll
    for (; scale > 1; scale >>= 1)
    {
        float4 enc = EvaluatePositionalEncoding(posf, iscale);
        outputArray[idx + 0] = FloatToInt8(enc.x, c_InputScale);
        outputArray[idx + 1] = FloatToInt8(enc.y, c_InputScale);
        outputArray[idx + 2] = FloatToInt8(enc.z, c_InputScale);
        outputArray[idx + 3] = FloatToInt8(enc.w, c_InputScale);

        idx += 4;
        iscale *= 2;
    }

    outputArray[idx + 0] = FloatToInt8(lod, c_InputScale);
    outputArray[idx + 1] = FloatToInt8(lod, c_InputScale);
    outputArray[idx + 2] = FloatToInt8(0.f, c_InputScale);
    outputArray[idx + 3] = FloatToInt8(0.f, c_InputScale);
}


template<class vecT>
__device__ __forceinline__
void EncodeSamplePosition_FP16( float2 posf, float lod, int offset, vecT& outputArray )
{
    int   idx    = offset;
    int   scale  = NTC_MLP_POS_ENC_SCALE;
    float iscale = 1.f / scale;

#pragma unroll
    for( ; scale > 1; scale >>= 1 )
    {
        float4 enc           = EvaluatePositionalEncoding( posf, iscale );
        outputArray[idx + 0] = half( enc.x );
        outputArray[idx + 1] = half( enc.y );
        outputArray[idx + 2] = half( enc.z );
        outputArray[idx + 3] = half( enc.w );

        idx += 4;
        iscale *= 2;
    }

    outputArray[idx + 0] = half( lod );
    outputArray[idx + 1] = half( lod );
}


template <int NETWORK_VERSION, class vecT>
__device__ __forceinline__
void PrepareNetworkInputs(
    const NtcTextureSetConstants& tsc,
    const uint8_t*                latents,
    float2                        uv,
    int2                          texel,
    int                           mipLevel,
    vecT&                         networkInputs )
{
    using Params = NTC_PARAMS<NETWORK_VERSION>;

    // Zero init the array - in some cases, OUTPUT_SIZE is rounded up from the actual used size.
    networkInputs = vecT( 0 );

    const NtcColorMipConstants& colorMip = tsc.colorMips[mipLevel];

    int inputOffset  = 0;
    int bufferOffset = 0;

    // Sample the latent grids
    SampleLatentGrid<Params::HR_FEATURES, true, vecT>(
        latents, bufferOffset, tsc.highResEncoding, tsc.highResNeuralMips[colorMip.neuralMip], uv, inputOffset, networkInputs );

    inputOffset += Params::SAMPLED_FEATURES_HR;

    SampleLatentGrid<Params::LR_FEATURES, false, vecT>(
        latents, bufferOffset, tsc.lowResEncoding, tsc.lowResNeuralMips[colorMip.neuralMip], uv, inputOffset, networkInputs );

    inputOffset += Params::SAMPLED_FEATURES_LR;

    // Encode the sample position
    EncodeSamplePosition<vecT>( make_float2( texel ) * colorMip.positionScale, colorMip.positionLod, inputOffset, networkInputs );
}


template<int NETWORK_VERSION, class vecT>
__device__ __forceinline__
void PrepareNetworkInputs_FP16(
    const NtcTextureSetConstants& constants,
    const uint8_t* latentsBuffer,
    float2 uv,
    int2 texel,
    int mipLevel,
    vecT& networkInputs)
{
    using Params = NTC_PARAMS<NETWORK_VERSION>;

    // Zero init the array - in some cases, OUTPUT_SIZE is rounded up from the actual used size.
    networkInputs = vecT(0);

   const NtcColorMipConstants& colorMip = constants.colorMips[mipLevel];

	int inputOffset  = 0;
    int bufferOffset = 0;

    // Sample the latent grids
    SampleLatentGrid_FP16<Params::HR_FEATURES, true, vecT>(
        latentsBuffer, bufferOffset, constants.highResEncoding, constants.highResNeuralMips[colorMip.neuralMip], uv, inputOffset, networkInputs );

    inputOffset += Params::SAMPLED_FEATURES_HR;

    SampleLatentGrid_FP16<Params::LR_FEATURES, false, vecT>(
        latentsBuffer, bufferOffset, constants.lowResEncoding, constants.lowResNeuralMips[colorMip.neuralMip], uv, inputOffset, networkInputs );

    inputOffset += Params::SAMPLED_FEATURES_LR;

    // Encode the sample position
    EncodeSamplePosition_FP16<vecT>(make_float2(texel) * colorMip.positionScale, colorMip.positionLod, inputOffset, networkInputs);
}


namespace HGELUClamp
{
    static __device__ constexpr float minval  = -3.f / 16.f;
    static __device__ constexpr float maxval  = 3.f;
    static __device__ constexpr int   bins    = 256;
    static __device__ constexpr float step    = ( maxval - minval ) / float( bins - 1 );
    static __device__ constexpr float invStep = 1.f / step;
    static __device__ constexpr int   qmax    = int( maxval / step );
    static __device__ constexpr int   qmin    = qmax - bins + 1;
    static __device__ constexpr int bias = -( bins / 2 ) - qmin;
};


template<class VecT>
__device__ __forceinline__
VecT activate(const VecT& x, bool scaleActivation)
{
    VecT tmp    = optixCoopVecFFMA( x, VecT( 1.0f / 3.0f ), VecT( 0.5f ) );
    tmp         = optixCoopVecMin( optixCoopVecMax( tmp, 0.0f ), 1.f );  // clamp(0,1)
    VecT result = optixCoopVecMin( x, 3.0f );
    result      = optixCoopVecMul( result, tmp );
    if( scaleActivation )
        result = optixCoopVecFFMA( result, VecT( (float)HGELUClamp::invStep ), VecT( (float)HGELUClamp::bias ) );
    return result;
}


template <class T_IN, int N_IN, class T_OUT, int N_OUT, bool DO_ACTIVATE>
__device__ __forceinline__
void EvaluateLayer_CoopVec(
    const OptixCoopVec<T_IN, N_IN>& inputArray,
    const uint8_t*                  weights,
    uint32_t                        weightOffset,
    uint32_t                        scaleOffset,
    uint32_t                        biasOffset,
    OptixCoopVec<T_OUT, N_OUT>&     outputArray )
{
    using T_VEC_IN = OptixCoopVec<T_IN, N_IN>;
    using T_VEC_OUT = OptixCoopVec<T_OUT, N_OUT>;

    OptixCoopVec<int32_t, N_OUT> z_i1 =
        optixCoopVecMatMul
        <
        OptixCoopVec<int32_t, N_OUT>,      // VecTOut
        OptixCoopVec<T_IN, N_IN>,          // VecTIn
        OPTIX_COOP_VEC_ELEM_TYPE_INT8,     // inputInterpretation
        MAT_LAYOUT,                        // matrixLayout
        false,                             // transpose
        N_OUT,                             // N
        N_IN,                              // K
        OPTIX_COOP_VEC_ELEM_TYPE_INT8,     // matrixElementType
        OPTIX_COOP_VEC_ELEM_TYPE_INT32     // biasElementType
        >(
            inputArray,                    // inputVector
            (CUdeviceptr)weights,          // matrix
            weightOffset,                  // matrixOffsetInBytes
            0,                             // bias
            0                              // biasOffsetInBytes
        );

    outputArray = optixCoopVecCvt<T_VEC_OUT>( z_i1 );

    T_VEC_OUT scale = optixCoopVecLoad<T_VEC_OUT>( weights + scaleOffset );
    T_VEC_OUT bias  = optixCoopVecLoad<T_VEC_OUT>( weights + biasOffset );

    outputArray = optixCoopVecFFMA( outputArray, scale, bias );

    if (DO_ACTIVATE) {
        outputArray = activate<T_VEC_OUT>(outputArray, true);
    }
}


template <class T_VEC_IN, class T_VEC_OUT>
__device__ __forceinline__
void EvaluateLayer_CoopVec_FP8(
    const T_VEC_IN& inputArray,
    const uint8_t*  weights,
    uint32_t        weightsOffsetInBytes,
    uint32_t&       biasOffsetInBytes,
    bool            scaleActivation,
    T_VEC_OUT&      outputArray )
{
    constexpr int N_IN  = T_VEC_IN::size;
    constexpr int N_OUT = T_VEC_OUT::size;

    outputArray =
        optixCoopVecMatMul
        <
        T_VEC_OUT,                            // VecTOut
        T_VEC_IN,                             // VecTIn
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3, // inputInterpretation
        MAT_LAYOUT,                           // matrixLayout
        false,                                // transpose
        N_OUT,                                // N
        N_IN,                                 // K
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3, // matrixElementType
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16      // biasElementType
        >(
            inputArray,                       // inputVector
            (CUdeviceptr)weights,             // matrix
            weightsOffsetInBytes,
            (CUdeviceptr)weights,             // bias
            biasOffsetInBytes
        );

    biasOffsetInBytes += T_VEC_OUT::size * sizeof( T_VEC_OUT::value_type );

    outputArray = activate<T_VEC_OUT>( outputArray, scaleActivation );
}


template <class T_VEC_IN, class T_VEC_OUT>
__device__ __forceinline__
void EvaluateOutputLayer_CoopVec_FP8(
    const T_VEC_IN& inputArray,
    const uint8_t*  weights,
    uint32_t        weightsOffsetInBytes,
    uint32_t&       scaleBiasOffsetInBytes,
    T_VEC_OUT&      outputArray )
{
    constexpr int N_IN  = T_VEC_IN::size;
    constexpr int N_OUT = T_VEC_OUT::size;
    using T_VEC_MAT_OUT = OptixCoopVec<int32_t, N_OUT>;

    T_VEC_MAT_OUT z_i1 =
        optixCoopVecMatMul
        <
        T_VEC_MAT_OUT,                     // VecTOut
        T_VEC_IN,                          // VecTIn
        OPTIX_COOP_VEC_ELEM_TYPE_INT8,     // inputInterpretation
        MAT_LAYOUT,                        // matrixLayout
        false,                             // transpose
        N_OUT,                             // N
        N_IN,                              // K
        OPTIX_COOP_VEC_ELEM_TYPE_INT8,     // matrixElementType
        OPTIX_COOP_VEC_ELEM_TYPE_INT32     // biasElementType
        >(
            inputArray,                    // inputVector
            (CUdeviceptr)weights,          // matrix
            weightsOffsetInBytes,
            0,                             // bias
            0                              // biasOffsetInBytes
        );

    outputArray = optixCoopVecCvt<T_VEC_OUT>( z_i1 );

    T_VEC_OUT scale = optixCoopVecLoad<T_VEC_OUT>( weights + scaleBiasOffsetInBytes );
    scaleBiasOffsetInBytes += N_OUT * sizeof( T_VEC_OUT::value_type );

    T_VEC_OUT bias = optixCoopVecLoad<T_VEC_OUT>( weights + scaleBiasOffsetInBytes );
    scaleBiasOffsetInBytes += N_OUT * sizeof( T_VEC_OUT::value_type );

    outputArray = optixCoopVecFFMA( outputArray, scale, bias );
}


__device__ __forceinline__ float fracf(float x) { return x - floorf(x); }

__device__ __forceinline__
float2 getStochasticTexelWeights(int2 imgSize, float2 uv, int2& outTopLeftPos)
{
    // modified version of SetupLatentBilinearFilter()
    float2       pixelPos   = uv * make_float2( imgSize ) - 0.5f;
    float2       topLeftPos = floor( pixelPos );
    const float2 d          = pixelPos - topLeftPos;
    outTopLeftPos           = make_int2( topLeftPos );
    return d;
}
 

template <class T_VEC_OUTPUT, int NETWORK_VERSION>
__device__ __forceinline__
bool inferTexelCoopVec_fp8( T_VEC_OUTPUT& outputLayer, NtcTextureSetConstants& tsc, uint8_t* latents, uint8_t* mlpWeights, int x, int y, int lod )
{
    using Params = NTC_PARAMS<NETWORK_VERSION>;

    const int    mipLevel  = lod;
    const int2   imageSize = GetTextureDimensions( tsc, mipLevel );
    const int2   texel     = make_int2( x, imageSize.y - 1 - y );
    const float2 uv        = ( make_float2( texel ) + 0.5f ) / make_float2( imageSize );

    constexpr int layerSize[5] = {
        Params::INPUT_CHANNELS,
        Params::HIDDEN_LAYER_CHANNELS,
        Params::HIDDEN_LAYER_CHANNELS,
        Params::HIDDEN_LAYER_CHANNELS,
        Params::OUTPUT_CHANNELS 
    };

    // For info about buffer layouts (mat weights, scales, biases), see NTC's TextureSet.cpp.
    // Here we are using NTC's GenericFP8 format, but we have converted the mats to inference_optimal layout 
    // see convertWeights() in optixNeuralTexture.cpp

    const int weightOffset0   = 0;
    const int weightOffset1   = weightOffset0 + int(optixCoopVecGetMatrixSize<layerSize[1], layerSize[0], OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3, MAT_LAYOUT>());
    const int weightOffset2   = weightOffset1 + int(optixCoopVecGetMatrixSize<layerSize[2], layerSize[1], OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3, MAT_LAYOUT>());
    const int weightOffset3   = weightOffset2 + int(optixCoopVecGetMatrixSize<layerSize[3], layerSize[2], OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3, MAT_LAYOUT>());
    const int weightOffsetEnd = weightOffset3 + int(optixCoopVecGetMatrixSize<layerSize[4], layerSize[3], OPTIX_COOP_VEC_ELEM_TYPE_INT8,        MAT_LAYOUT>());

    using T_VEC_INPUT  = OptixCoopVec<half, Params::INPUT_CHANNELS>;
    using T_VEC_HIDDEN = OptixCoopVec<half, Params::HIDDEN_LAYER_CHANNELS>;

    T_VEC_INPUT  networkInputs;
    T_VEC_HIDDEN hiddenOutput1;
    T_VEC_HIDDEN hiddenOutput2;
    T_VEC_HIDDEN hiddenOutput3;

    PrepareNetworkInputs_FP16<NETWORK_VERSION, T_VEC_INPUT>(tsc, latents, uv, texel, mipLevel, networkInputs);

    uint32_t scaleBiasOffsetInBytes = weightOffsetEnd;

    EvaluateLayer_CoopVec_FP8<T_VEC_INPUT, T_VEC_HIDDEN>(
        networkInputs, mlpWeights, weightOffset0, scaleBiasOffsetInBytes, false, hiddenOutput1);

    EvaluateLayer_CoopVec_FP8<T_VEC_HIDDEN, T_VEC_HIDDEN>(
        hiddenOutput1, mlpWeights, weightOffset1, scaleBiasOffsetInBytes, false, hiddenOutput2);

    EvaluateLayer_CoopVec_FP8<T_VEC_HIDDEN, T_VEC_HIDDEN>(
        hiddenOutput2, mlpWeights, weightOffset2, scaleBiasOffsetInBytes, true, hiddenOutput3);

    EvaluateOutputLayer_CoopVec_FP8<T_VEC_HIDDEN, T_VEC_OUTPUT>(
        hiddenOutput3, mlpWeights, weightOffset3, scaleBiasOffsetInBytes, outputLayer);

    return true;
}


template <class T_VEC_OUTPUT, int NETWORK_VERSION> 
__device__ __forceinline__
bool inferTexelCoopVec_int8( T_VEC_OUTPUT& outputLayer, NtcTextureSetConstants& tsc, uint8_t* latents, uint8_t* mlpWeights, int x, int y, int lod )
{
    using Params = NTC_PARAMS<NETWORK_VERSION>;

    const int    mipLevel  = lod;
    const int2   imageSize = GetTextureDimensions( tsc, mipLevel );
    const int2   texel     = make_int2( x, imageSize.y - 1 - y );
    const float2 uv        = ( make_float2( texel ) + 0.5f ) / make_float2( imageSize );

    constexpr int layerSize[5] = {
        Params::INPUT_CHANNELS,
        Params::HIDDEN_LAYER_CHANNELS,
        Params::HIDDEN_LAYER_CHANNELS,
        Params::HIDDEN_LAYER_CHANNELS,
        Params::OUTPUT_CHANNELS };

    // The matrix weights for all layers are stored in Int8 format and densely packed one after another. 
    // Then the scale vectors for all layers are stored in Float32 format and are densely packed one after another. 
    // Finally, the bias vectors for all layers are stored in Float32 format and also densely packed.

    int weightSize[4] = {
        int( optixCoopVecGetMatrixSize<layerSize[1], layerSize[0], OPTIX_COOP_VEC_ELEM_TYPE_INT8, MAT_LAYOUT>() ),
        int( optixCoopVecGetMatrixSize<layerSize[2], layerSize[1], OPTIX_COOP_VEC_ELEM_TYPE_INT8, MAT_LAYOUT>() ),
        int( optixCoopVecGetMatrixSize<layerSize[3], layerSize[2], OPTIX_COOP_VEC_ELEM_TYPE_INT8, MAT_LAYOUT>() ),
        int( optixCoopVecGetMatrixSize<layerSize[4], layerSize[3], OPTIX_COOP_VEC_ELEM_TYPE_INT8, MAT_LAYOUT>() ) };

    // Compute offsets for weights, scales, and biases. Compute pointers for scales and biases
    // We skip pointer for weights because it's preferable to to the same weights pointer 
    // with a different offset for successive matmul calls

    constexpr int numWeights                   = 4;
    uint32_t      weightOffset[numWeights + 1] = {};
    for( int i = 1; i < numWeights + 1; i++ )
        weightOffset[i] = weightOffset[i - 1] + weightSize[i - 1];

    constexpr int numScales                  = 4;
    int           scaleOffset[numScales + 1] = {};
    scaleOffset[0]                           = weightOffset[numWeights];
    // NB subtle point warning: layerSize is for the OUTPUT layer here
    for( int i = 1; i < numScales + 1; i++ )
        scaleOffset[i] = scaleOffset[i - 1] + layerSize[i] * sizeof( float );

    const int numBiases                 = 4;
    int       biasOffset[numBiases + 1] = {};
    biasOffset[0]                       = scaleOffset[numScales];
    // NB subtle point warning: layerSize is for the OUTPUT layer here
    for( int i = 1; i < numBiases + 1; i++ )
        biasOffset[i] = biasOffset[i - 1] + layerSize[i] * sizeof( float );

    using T_VEC_INPUT  = OptixCoopVec<int8_t, Params::INPUT_CHANNELS>;
    using T_VEC_HIDDEN = OptixCoopVec<float, Params::HIDDEN_LAYER_CHANNELS>;

    T_VEC_INPUT  networkInputs;
    T_VEC_HIDDEN hiddenOutput1;
    T_VEC_HIDDEN hiddenOutput2;
    T_VEC_HIDDEN hiddenOutput3;

    PrepareNetworkInputs<NETWORK_VERSION, T_VEC_INPUT>(tsc, latents, uv, texel, mipLevel, networkInputs);

    EvaluateLayer_CoopVec<int8_t, Params::INPUT_CHANNELS, float, Params::HIDDEN_LAYER_CHANNELS, true>(
        networkInputs, mlpWeights, weightOffset[0], scaleOffset[0], biasOffset[0], hiddenOutput1);

    EvaluateLayer_CoopVec<float, Params::HIDDEN_LAYER_CHANNELS, float, Params::HIDDEN_LAYER_CHANNELS, true>(
        hiddenOutput1, mlpWeights, weightOffset[1], scaleOffset[1], biasOffset[1], hiddenOutput2);

    EvaluateLayer_CoopVec<float, Params::HIDDEN_LAYER_CHANNELS, float, Params::HIDDEN_LAYER_CHANNELS, true>(
        hiddenOutput2, mlpWeights, weightOffset[2], scaleOffset[2], biasOffset[2], hiddenOutput3);

    EvaluateLayer_CoopVec<float, Params::HIDDEN_LAYER_CHANNELS, float, Params::OUTPUT_CHANNELS, false>(
        hiddenOutput3, mlpWeights, weightOffset[3], scaleOffset[3], biasOffset[3], outputLayer);

    return true;
}
