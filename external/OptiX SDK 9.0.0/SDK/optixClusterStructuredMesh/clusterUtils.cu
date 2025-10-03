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

#include "clusterUtils.h"


//////////////////////////////////////////////////////////////////////////////////
// helper functions for generating vertex and index buffer                      //
//////////////////////////////////////////////////////////////////////////////////

// Hashing function (used for fast on-device pseudorandom numbers for randomness in noise)
__forceinline__ __device__ float hash( float2 st )
{
    const float result = sin( dot( st, make_float2( 12.9898, 78.233 ) ) ) * 43758.5453123;
    return result - floor( result );
}


// 2D Noise based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
__forceinline__ __device__ float noise( float2 st )
{
    const float2 i = floor( st );
    const float2 f = st - i;

    // Four corners in 2D of a tile
    const float a = hash( i );
    const float b = hash( i + make_float2( 1.0, 0.0 ) );
    const float c = hash( i + make_float2( 0.0, 1.0 ) );
    const float d = hash( i + make_float2( 1.0, 1.0 ) );

    // Cubic Hermine Curve. Same as SmoothStep()
    const float2 u = f * f * ( 3.0 - 2.0 * f );

    // Mix 4 coorners percentages
    return a + u.x * ( b - a ) + ( c - a ) * u.y * ( 1.0 - u.x ) + ( d - b ) * u.x * u.y;
}


__forceinline__ __device__ float3 deform_vertex( const float3 c, float3 mousePosInWorld, float time )
{
    // Apply sine wave heigh field to the y coordinate
    const float wavePhase =
        0.2f * noise( make_float2( c.x, c.z ) * 3.f + time ) + 0.3f * noise( make_float2( c.z, c.x ) * 2.f - time * 0.5f );
    const float mousePhase =
        clamp( length( make_float2( c.x - mousePosInWorld.x, c.z - mousePosInWorld.z ) ) * M_PIf * 3.f, 0.f, M_PIf );
    return make_float3( c.x, c.y + wavePhase + 0.15f * cosf( mousePhase ), c.z );
}


__global__ void generate_vertices_cluster( const ClusterTessConfig tessConfig, float3* outVertices )
{
    const uint32_t vtxId                 = blockIdx.x * blockDim.x + threadIdx.x;
    const uint2    gridDims              = tessConfig.gridDims;
    const uint32_t numClusters           = gridDims.x * gridDims.y;
    const uint2    cVtxSize              = make_uint2( tessConfig.clusterSize.x + 1, tessConfig.clusterSize.y + 1 );
    const uint32_t numVerticesPerCluster = cVtxSize.x * cVtxSize.y;

    if( vtxId >= numClusters * numVerticesPerCluster )
        return;

    const uint32_t clusterId    = vtxId / numVerticesPerCluster;
    const uint32_t vertexOffset = clusterId * numVerticesPerCluster;

    const uint2 uv_ccll = make_uint2( clusterId % gridDims.x, clusterId / gridDims.x );
    const uint2 uv_ccur = uv_ccll + make_uint2( 1 );

    const float2 uv_ccll_norm = make_float2( (float)uv_ccll.x / gridDims.x, (float)uv_ccll.y / gridDims.y );
    const float2 uv_ccur_norm = make_float2( (float)uv_ccur.x / gridDims.x, (float)uv_ccur.y / gridDims.y );

    const float3 gc_ll = make_float3( -1.f, 0.f, -1.f );  // lower-left corner vtx of the grid
    const float3 gc_ur = make_float3( 1.f, 0.f, 1.f );    // up-right corner vtx of the grid

    const float3 cc_ll = gc_ll + ( gc_ur - gc_ll ) * make_float3( uv_ccll_norm.x, 0.f, uv_ccll_norm.y );  // lower-left corner vtx of the cluster
    const float3 cc_ur = gc_ll + ( gc_ur - gc_ll ) * make_float3( uv_ccur_norm.x, 0.f, uv_ccur_norm.y );  // up-right corner vtx of the cluster

    const int x = ( vtxId - vertexOffset ) % cVtxSize.x;
    const int y = ( vtxId - vertexOffset ) / cVtxSize.x;

    const float xNorm = (float)x / tessConfig.clusterSize.x;
    const float yNorm = (float)y / tessConfig.clusterSize.y;

    const float3 v     = cc_ll + ( cc_ur - cc_ll ) * make_float3( xNorm, 0, yNorm );
    outVertices[vtxId] = deform_vertex( v, tessConfig.mousePosInWorld, tessConfig.animationTime );
}


__global__ void generate_indices_cluster( const ClusterTessConfig tessConfig, uint3* outIndices )
{
    const uint32_t quadId             = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t numClusters        = tessConfig.gridDims.x * tessConfig.gridDims.y;
    const uint32_t numQuadsPerCluster = tessConfig.clusterSize.x * tessConfig.clusterSize.y;

    if( quadId >= numClusters * numQuadsPerCluster )
        return;

    const uint32_t clusterId       = quadId / numQuadsPerCluster;
    const uint32_t quadIdInCluster = quadId % numQuadsPerCluster;

    const uint2    cVtxSize              = make_uint2( tessConfig.clusterSize.x + 1, tessConfig.clusterSize.y + 1 );
    const uint32_t numVerticesPerCluster = cVtxSize.x * cVtxSize.y;
    const uint32_t vertexOffset          = clusterId * numVerticesPerCluster;

    const uint32_t u = quadIdInCluster % tessConfig.clusterSize.x;
    const uint32_t v = quadIdInCluster / tessConfig.clusterSize.x;

    uint3 localIndice_0 = make_uint3( v * cVtxSize.x + u, v * cVtxSize.x + u + 1, ( v + 1 ) * cVtxSize.x + u );
    uint3 localIndice_1 = make_uint3( v * cVtxSize.x + u + 1, ( v + 1 ) * cVtxSize.x + u + 1, ( v + 1 ) * cVtxSize.x + u );

    outIndices[quadId * 2 + 0] = make_uint3( vertexOffset ) + localIndice_0;
    outIndices[quadId * 2 + 1] = make_uint3( vertexOffset ) + localIndice_1;
}


// vertices are duplicated at the cluster boundary
void generateAnimatedVertices( CUstream& stream, const ClusterTessConfig tessConfig, CUdeviceptr& d_outVertices)
{
    const uint32_t numClusters           = tessConfig.gridDims.x * tessConfig.gridDims.y;
    const uint32_t numVerticesPerCluster = ( tessConfig.clusterSize.x + 1 ) * ( tessConfig.clusterSize.y + 1 );
    const uint32_t numVertices           = numClusters * numVerticesPerCluster;

    dim3 threadsPerBlock( 32, 1 );
    int  numBlocks = ( numVertices + threadsPerBlock.x - 1 ) / threadsPerBlock.x;
    generate_vertices_cluster<<<numBlocks, threadsPerBlock, 0, stream>>>( tessConfig, reinterpret_cast<float3*>( d_outVertices ) );
}


void generateIndices( CUstream& stream, const ClusterTessConfig tessConfig, CUdeviceptr& d_outIndices)
{
    const uint32_t clusterCount = tessConfig.gridDims.x * tessConfig.gridDims.y;
    const uint32_t numQuads     = clusterCount * tessConfig.clusterSize.x * tessConfig.clusterSize.y;

    dim3 threadsPerBlock( 32, 1 );
    int  numBlocks = ( numQuads + threadsPerBlock.x - 1 ) / threadsPerBlock.x;
    generate_indices_cluster<<<numBlocks, threadsPerBlock, 0, stream>>>( tessConfig, reinterpret_cast<uint3*>( d_outIndices ) );
}


//////////////////////////////////////////////////////////////////////////////////
// helper functions for building templates, CLAS, and BLAS                      //
//////////////////////////////////////////////////////////////////////////////////
__global__ void calculateClasOutputBufferSizeAndOffsets_kernel( const uint32_t      clusterCount,
                                                                const uint32_t      maxClusterEdgeSegments,
                                                                const uchar2*       edgeSegments,
                                                                const uint32_t*     templateSizes,
                                                                const size_t        strideInBytes,
                                                                size_t*             d_clasAddressOffsets,
                                                                unsigned long long* clusterMemorySize )
{
    const uint32_t clusterId = blockIdx.x * blockDim.y + threadIdx.y;
    if( clusterId >= clusterCount )
        return;

    const uchar2 es =
        *reinterpret_cast<const uchar2*>( reinterpret_cast<const unsigned char*>( edgeSegments ) + clusterId * strideInBytes );
    assert( es.x <= maxClusterEdgeSegments && es.y <= maxClusterEdgeSegments );

    const uint32_t templateIndex    = ( es.x - 1 ) * maxClusterEdgeSegments + ( es.y - 1 );
    d_clasAddressOffsets[clusterId] = atomicAdd( clusterMemorySize, (unsigned long long)templateSizes[templateIndex] );
}


void calculateClasOutputBufferSizeAndOffsets( CUstream&       stream,
                                              const uint32_t  clusterCount,
                                              const uint32_t  g_maxClusterEdgeSegments,
                                              const uchar2*   d_edgeSegments,
                                              const uint32_t* d_templateSizes,
                                              const size_t    strideInBytes,
                                              size_t*         d_outputSizeInBytes,
                                              size_t*         d_clasAddressOffsets )
{
    calculateClasOutputBufferSizeAndOffsets_kernel<<<clusterCount, 1, 0, stream>>>(
        clusterCount, g_maxClusterEdgeSegments, d_edgeSegments, d_templateSizes, strideInBytes, d_clasAddressOffsets,
        reinterpret_cast<unsigned long long*>( d_outputSizeInBytes ) );
}

__global__ void assign_explicit_addresses_kernel( const uint32_t    clusterCount,
                                                  const size_t*     d_clasAddressOffsets,
                                                  const CUdeviceptr d_clasBuffer,
                                                  CUdeviceptr*      d_clasPtrsBuffer )
{
    const uint32_t clusterId = blockIdx.x * blockDim.y + threadIdx.y;
    if( clusterId >= clusterCount )
        return;

    d_clasPtrsBuffer[clusterId] = d_clasBuffer + d_clasAddressOffsets[clusterId];
}


void assignExplicitAddresses( CUstream&         stream,
                              const uint32_t    clusterCount,
                              const size_t*     d_clasAddressOffsets,
                              const CUdeviceptr d_clasBuffer,
                              CUdeviceptr*      d_clasPtrsBuffer )
{
    assign_explicit_addresses_kernel<<<clusterCount, 1, 0, stream>>>( clusterCount, d_clasAddressOffsets, d_clasBuffer, d_clasPtrsBuffer );
}


__global__ void makeClusters_kernel( const ClusterTessConfig tessConfig, Cluster* d_outClusters )
{
    const uint32_t clusterId    = blockIdx.x * blockDim.x + threadIdx.x;
    const uint2    gridDim      = tessConfig.gridDims;
    const uint32_t clusterCount = gridDim.x * gridDim.y;

    if( clusterId >= clusterCount )
        return;

    const uint2 edgeSegments = tessConfig.clusterSize;

    d_outClusters[clusterId].size         = make_uchar2( edgeSegments.x, edgeSegments.y );
    d_outClusters[clusterId].vertexOffset = ( edgeSegments.x + 1 ) * ( edgeSegments.y + 1 ) * clusterId;
}


// tesselllate a square surface into gridDim.x * gridDim.y clusters and assign the cluster size and vertexOffset
void makeClusters( CUstream& stream, const ClusterTessConfig tessConfig, Cluster* d_outClusters )
{
    const uint32_t clusterCount = tessConfig.gridDims.x * tessConfig.gridDims.y;

    dim3 blockSize( 32, 1, 1 );
    dim3 gridSize( ( clusterCount + blockSize.x - 1 ) / blockSize.x, 1, 1 );
    makeClusters_kernel<<<gridSize, blockSize, 0, stream>>>( tessConfig, d_outClusters );
}


__global__ void makeTemplatesArgsDataForGetSizes_kernel( const CUdeviceptr*                        d_templateAddresses,
                                                         const uint32_t                            numTemplates,
                                                         OptixClusterAccelBuildInputTemplatesArgs* d_templatesArgs )
{
    const uint32_t templateId = blockIdx.x * blockDim.x + threadIdx.x;
    if( templateId >= numTemplates )
        return;

    OptixClusterAccelBuildInputTemplatesArgs args = { 0 };
    args.clusterTemplate                          = d_templateAddresses[templateId];

    d_templatesArgs[templateId] = args;
}

// Instantiate Templates GetSizes, used in initStructuredClusterTemplates()
void makeTemplatesArgsDataForGetSizes( CUstream&                                 stream,
                                       const CUdeviceptr*                        d_templateAddresses,
                                       const uint32_t                            numTemplates,
                                       OptixClusterAccelBuildInputTemplatesArgs* d_templatesArgs )
{
    dim3 blockSize( 32, 1, 1 );
    dim3 gridSize( ( numTemplates + blockSize.x - 1 ) / blockSize.x, 1, 1 );
    makeTemplatesArgsDataForGetSizes_kernel<<<gridSize, blockSize, 0, stream>>>( d_templateAddresses, numTemplates, d_templatesArgs );
}


__global__ void makeTemplatesArgs_kernel( const Cluster*                            d_clusters,
                                          const size_t                              clusterCount,
                                          const uint32_t                            maxClusterEdgeSegments,
                                          const CUdeviceptr*                        d_templateAddresses,
                                          const float3*                             d_vertexPositions,
                                          OptixClusterAccelBuildInputTemplatesArgs* d_templatesArgs )
{
    const uint32_t clusterId = blockIdx.x * blockDim.x + threadIdx.x;
    if( clusterId >= clusterCount )
        return;

    const Cluster     cluster               = d_clusters[clusterId];
    const CUdeviceptr d_vertexBufferAddress = reinterpret_cast<CUdeviceptr>( &d_vertexPositions[cluster.vertexOffset] );
    const uint32_t    templateIndex         = ( cluster.size.x - 1 ) * maxClusterEdgeSegments + ( cluster.size.y - 1 );

    OptixClusterAccelBuildInputTemplatesArgs args = { 0 };
    args.clusterIdOffset                          = clusterId;
    args.clusterTemplate                          = d_templateAddresses[templateIndex];
    args.vertexBuffer                             = d_vertexBufferAddress;
    args.vertexStrideInBytes                      = sizeof( float3 );

    d_templatesArgs[clusterId] = args;
}


void makeTemplatesArgsData( CUstream&                                 stream,
                            const Cluster*                            d_clusters,
                            const uint32_t                            clusterCount,
                            const uint32_t                            g_maxClusterEdgeSegments,
                            const CUdeviceptr*                        d_templateAddresses,
                            const float3*                             d_vertexPositions,
                            OptixClusterAccelBuildInputTemplatesArgs* d_templatesArgs )
{
    dim3 blockSize( 32, 1, 1 );
    dim3 gridSize( ( clusterCount + blockSize.x - 1 ) / blockSize.x, 1, 1 );
    makeTemplatesArgs_kernel<<<gridSize, blockSize, 0, stream>>>( d_clusters, clusterCount, g_maxClusterEdgeSegments,
                                                                  d_templateAddresses, d_vertexPositions, d_templatesArgs );
}


__global__ void makeClustersArgs_kernel( const size_t*                            d_clusterOffsets,
                                         const uint32_t                           instanceCount,
                                         const CUdeviceptr*                       d_clasPtrs,
                                         OptixClusterAccelBuildInputClustersArgs* d_clustersArgs )
{
    // Transform cluster offsets into indirect args
    const uint32_t instanceId = blockIdx.x * blockDim.x + threadIdx.x;
    if( instanceId >= instanceCount )
        return;

    OptixClusterAccelBuildInputClustersArgs args = { 0 };
    args.clusterHandlesCount                     = (uint32_t)( d_clusterOffsets[instanceId + 1] - d_clusterOffsets[instanceId] );
    args.clusterHandlesBuffer                    = CUdeviceptr( d_clasPtrs + d_clusterOffsets[instanceId] );

    d_clustersArgs[instanceId] = args;
}


void makeClustersArgsData( CUstream&                                stream,
                           const size_t*                            d_clusterOffsets,
                           const uint32_t                           instanceCount,
                           const CUdeviceptr*                       d_clasPtrs,
                           OptixClusterAccelBuildInputClustersArgs* d_clustersArgs )
{
    dim3 blockSize( 32, 1, 1 );
    dim3 gridSize( ( instanceCount + blockSize.x - 1 ) / blockSize.x, 1, 1 );
    makeClustersArgs_kernel<<<gridSize, blockSize, 0, stream>>>( d_clusterOffsets, instanceCount, d_clasPtrs, d_clustersArgs );
}


__global__ void copyGasHandlesToInstanceArray_kernel( OptixInstance* d_instances, const CUdeviceptr* d_gasHandles, uint32_t instanceCount )
{
    const uint32_t instanceId = blockIdx.x * blockDim.x + threadIdx.x;
    if( instanceId >= instanceCount )
        return;

    d_instances[instanceId].traversableHandle = d_gasHandles[instanceId];
}

void copyGasHandlesToInstanceArray( CUstream& stream, OptixInstance* d_instances, const CUdeviceptr* d_gasHandles, uint32_t instanceCount )
{
    dim3 blockSize( 32, 1, 1 );
    dim3 gridSize( ( instanceCount + blockSize.x - 1 ) / blockSize.x, 1, 1 );
    copyGasHandlesToInstanceArray_kernel<<<gridSize, blockSize, 0, stream>>>( d_instances, d_gasHandles, instanceCount );
}