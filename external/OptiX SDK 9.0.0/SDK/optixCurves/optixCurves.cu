/*

 * SPDX-FileCopyrightText: Copyright (c) 2019 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <optix.h>

#include "optixCurves.h"
#include <cuda/helpers.h>
#include <random.h>

#include <sutil/vec_math.h>

extern "C" {
__constant__ Params params;
}

// ROCAPS_TUNE_CURVE_PARAMETER encloses the example code for rocaps curve parameter tuning. 
// #define ROCAPS_TUNE_CURVE_PARAMETER

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_uint( p.x ) );
    optixSetPayload_1( __float_as_uint( p.y ) );
    optixSetPayload_2( __float_as_uint( p.z ) );
}


static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
{
    const float3 U = params.cam_u;
    const float3 V = params.cam_v;
    const float3 W = params.cam_w;
    const float2 d = 2.0f * make_float2(
            static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
            static_cast<float>( idx.y ) / static_cast<float>( dim.y )
            ) - 1.0f;

    origin    = params.cam_eye;
    direction = normalize( d.x * U + d.y * V + W );
}


extern "C" __global__ void __raygen__basic()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay( idx, dim, ray_origin, ray_direction );

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
            params.handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            1e16f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2 );
    float3 result;
    result.x = __uint_as_float( p0 );
    result.y = __uint_as_float( p1 );
    result.z = __uint_as_float( p2 );

    // Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_color( result );
}


extern "C" __global__ void __raygen__motion_blur()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen
    float3 ray_origin, ray_direction;
    computeRay( idx, dim, ray_origin, ray_direction );

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    const int NUM_SAMPLES = 100;
    float3 result = {};
    unsigned int seed = tea<4>(idx.y * dim.y + dim.x, idx.x);
    for( int i = 0; i < NUM_SAMPLES; ++i )
    {
        const float ray_time = rnd(seed); // compute next random ray time in [0, 1[
        optixTrace( params.handle, ray_origin, ray_direction,
                    0.0f,                        // Min intersection distance
                    1e16f,                       // Max intersection distance
                    ray_time,                    // rayTime -- used for motion blur
                    OptixVisibilityMask( 255 ),  // Specify always visible
                    OPTIX_RAY_FLAG_NONE,
                    0,  // SBT offset   -- See SBT discussion
                    1,  // SBT stride   -- See SBT discussion
                    0,  // missSBTIndex -- See SBT discussion
                    p0, p1, p2 );
        result.x += __uint_as_float( p0 );
        result.y += __uint_as_float( p1 );
        result.z += __uint_as_float( p2 );
    }

    // Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_color( result / NUM_SAMPLES );
}


extern "C" __global__ void __miss__ms()
{
    MissData* miss_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    setPayload(  miss_data->bg_color );
}

#ifdef ROCAPS_TUNE_CURVE_PARAMETER

static __forceinline__ __device__ void convertToDifferentialBezier( OptixPrimitiveType primType, float4 q[4], float4 p[4] )
{
    if( primType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE ||
        primType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS )
    {
        p[0] = q[0] * 0.5f + q[1] * 0.5f;
        p[1] = q[1] - q[0];
        p[2] = q[0] * 0.5f + q[2] * 0.5f - q[1];
    }
    else if( primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE ||
             primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE_ROCAPS )
    {
        p[0] = q[0] * (float)(1./6.) + q[1] * (float)(4./6.) + q[2] * (float)(1./6.);
        p[1] = q[2] - q[0];
        p[2] = q[2] - q[1];
        p[3] = q[3] - q[1];
    }
    else if( primType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM ||
             primType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM_ROCAPS )
    {
        p[0] = q[1];
        p[1] = q[2] - q[0];
        p[2] = q[0] * 0.5f + q[1] * ( -2.5f ) + q[2] * 2.5f + q[3] * ( -0.5f );
        p[3] = q[3] - q[1];
    }
    else  // OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER ||
          // OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER_ROCAPS
    {
        p[0] = q[0];
        p[1] = 6.f * q[1] - 6.f * q[0];
        p[2] = 3.f * q[2] - 3.f * q[1];
        p[3] = 6.f * q[3] - 6.f * q[2];
    }
}

static __forceinline__ __device__ float L1norm( const float4& v )
{
    return fabsf( v.x ) + fabsf( v.y ) + fabsf( v.z );
}

static __forceinline__ __device__ float3 terms( float u )
{
    float uu = u * u;
    float u3 = (float)(1./6.) * uu * u;
    return make_float3( u3 + 0.5f * ( u - uu ), uu - 4.f * u3, u3 );
}

template <int degree = 3>
static __forceinline__ __device__ void capsuleTune( float4* p, float3 raypos, float3 raydir, float& u )
{
    float us = u;
    float u1, f0, f1;

    float c = dot( raydir, raydir );
    float h = 0;  // u1 = u0 at first iteration

    // cover 3 * 0.1 = 30% of the capsule length
    for( int nrep = 0; nrep <= 3; nrep++ )
    {
        u1 = u + h;
        // f1 = capsule_error at u1,
        // dot(given raypos - adjusted_curve_pos, velocity)
        float4 c04, c14;
        if( degree == 2 )
        {
            c04 = p[0] + u1 * p[1] + u1 * u1 * p[2];  // position( u )
            c14 = p[1] + 2.f * u1 * p[2];              // velocity( u );
        }
        else
        {
            float3 t   = terms( u1 );
            c04 = p[0] + t.x * p[1] + t.y * p[2] + t.z * p[3];  // position( u );
            float  v   = 1.f - u1;
            c14 = 0.5f * v * v * p[1] + 2.f * v * u1 * p[2] + 0.5f * u1 * u1 * p[3];  // velocity( u );
        }
        float3 c0  = make_float3( c04 );
        float3 c1  = make_float3( c14 );
        float  r0  = c04.w;  // radius
        float  r1  = c14.w;  // its derivative
        f1         = dot( raypos - c0, c1 ) + r0 * r1;

        if( h == 0 )
        {
            // Compute step size.
            const float minh = 0.02f, maxh = 0.25f;
            float vel = L1norm( p[1] );  //velocity_a();
            // du ~ (smallest) curvature radius
            float du;
            if( degree == 2 )
            {
                float acceleration_a = L1norm( p[2] );
                du                   = vel / acceleration_a;
            }
            else
            {
                float acceleration_a = L1norm( 2.f * p[2] - p[1] );
                float jerk_a         = L1norm( p[1] + p[3] - 4.f * p[2] ) * (float)(1./3.);
                du                   = vel / ( acceleration_a + jerk_a );
            }
            du = minh * ( 1.f + du );
            du *= clamp( c / vel, 0.5f, 3.0f );
            // make step ~ primitive's footprint on the screen
            float avg_radius = degree == 2 ? p[0].w + p[1].w * 0.5f   + p[2].w * (float)(1./3.):
                                             p[0].w + p[1].w * 0.125f + p[2].w * (float)(1./6.) + p[3].w * (float)(1./24.);
            du               = fminf( du, 10 * (5 - degree) * avg_radius / vel );  // ~10 diameters
            float step       = fminf( du + 0.1f * minh, maxh );     // [0 + 0.1f * minh, maxh]

            h = 0.1f * copysignf( step, f1 );
        }
        else
        {
            if( f1 > 0 != f0 > 0 )
            {
                // intercept[{u0,u1}, {f0,f1}]
                u = ( f0 * u1 - f1 * u ) / ( f0 - f1 );
                break;
            }
            if( fabsf( f1 ) > fabsf( f0 ) )
            {
                u = us;  // something is wrong, restore the original solution
            }
        }
        u  = u1;
        f0 = f1;
    }
    u = u < 0 ? 0 : u > 1 ? 1 : u;
}

static __forceinline__ __device__ void tuneCurveParameter( float&  u ) 
{
    OptixPrimitiveType primType = optixGetPrimitiveType();

    bool hitcap = ( u == 0 ) || ( u == 1 );
    bool rocaps = ( primType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS ||
                    primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE_ROCAPS     ||
                    primType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM_ROCAPS        ||
                    primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER_ROCAPS );
    if( hitcap || !rocaps )
        return;

    // Get vertex data of curve segment.
    float4 q[4];
    switch( primType )
    {
        case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS:
            optixGetQuadraticBSplineRocapsVertexData( q );
            break;
        case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE_ROCAPS:
            optixGetCubicBSplineRocapsVertexData( q );
            break;
        case OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM_ROCAPS:
            optixGetCatmullRomRocapsVertexData( q );
            break;
        case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER_ROCAPS:
            optixGetCubicBezierRocapsVertexData( q );
            break;
    }

    // Differential Bezier representation of the curve segment.
    float4 p[4];
    convertToDifferentialBezier( primType, q, p );

    const float  t_hit        = optixGetRayTmax();
    const float3 obj_ray_orig = optixTransformPointFromWorldToObjectSpace( optixGetWorldRayOrigin() );
    const float3 obj_ray_dir  = optixTransformVectorFromWorldToObjectSpace( optixGetWorldRayDirection() );
    const float3 obj_raypos   = obj_ray_orig + t_hit * obj_ray_dir;

    if( primType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS )
        capsuleTune<2>( p, obj_raypos, obj_ray_dir, u );
    else  // cubic type
        capsuleTune<3>( p, obj_raypos, obj_ray_dir, u );

    return;
}

#endif // ROCAPS_TUNE_CURVE_PARAMETER

extern "C" __global__ void __closesthit__ch()
{
    // When built-in curve intersection is used, the curve parameter u is provided
    // by the OptiX API. The parameter’s range is [0,1] over the curve segment,
    // with u=0 or u=1 only on the end caps.
    float u = optixGetCurveParameter();

#ifdef ROCAPS_TUNE_CURVE_PARAMETER

    // The rocaps intersector provides the curve parameter of the intersection,
    // where its accuracy depends on the curve shape.
    // If the curve parameter is used directly, for example, in normal computations,
    // it can happen that its accuracy is insufficient on rapidly varying surfaces.
    // The curve parameter can be fine-tuned by using the following tuneCurveParameter routine.

    tuneCurveParameter( u );

    // For normal computation see computeNormal in optixHair.cu in the optixHair sample.

#endif // ROCAPS_TUNE_CURVE_PARAMETER

    // linearly interpolate from black to orange
    setPayload( make_float3( u, u / 3.0f, 0.0f ) );
}
