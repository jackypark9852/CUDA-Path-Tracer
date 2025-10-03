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

#include "unstructuredClusterUtils.h"


__global__ void deformClusterVertices_kernel( const uint32_t clusterCount, const float animationTime, Cluster* d_clusters )
{
    const uint32_t clusterId = blockIdx.x * blockDim.y + threadIdx.y;
    if( clusterId >= clusterCount )
        return;

    const Cluster& cluster  = d_clusters[clusterId];
    float3*        vertices = reinterpret_cast<float3*>( cluster.d_positions );
    for( uint32_t vtxIdx = threadIdx.x; vtxIdx < cluster.vertexCount; vtxIdx += 32 )
    {
        float3& vtx = vertices[vtxIdx];
        float   r   = M_PIf * length( make_float2( vtx.x, vtx.z ) );
        vtx.y += 0.1f * cosf( 4.f * ( 0.0015 * r + animationTime ) );
    }
}


void deformClusterVertices( CUstream& stream, const uint32_t clusterCount, const float animationTime, Cluster* d_clusters )
{
    dim3 blockSize( 32, 32, 1 );
    dim3 gridSize( ( clusterCount + blockSize.y - 1 ) / blockSize.y, 1, 1 );
    deformClusterVertices_kernel<<<gridSize, blockSize, 0, stream>>>( clusterCount, animationTime, d_clusters );
}


//////////////////////////////////////////////////////////////////////////////////
// helper functions for building templates, CLAS, and GAS                       //
//////////////////////////////////////////////////////////////////////////////////
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


__global__ void calculateClasOutputBufferSizeAndOffsets_kernel( const uint32_t      clusterCount,
                                                                const uint32_t*     templateSizes,
                                                                size_t*             d_clasAddressOffsets,
                                                                unsigned long long* clusterMemorySize )
{
    const uint32_t clusterId = blockIdx.x * blockDim.y + threadIdx.y;
    if( clusterId >= clusterCount )
        return;

    // one unique template per cluster
    d_clasAddressOffsets[clusterId] = atomicAdd( clusterMemorySize, (unsigned long long)templateSizes[clusterId] );
}


void calculateClasOutputBufferSizeAndOffsets( CUstream&       stream,
                                              const uint32_t  clusterCount,
                                              const uint32_t* d_templateSizes,
                                              size_t*         d_outputSizeInBytes,
                                              size_t*         d_clasAddressOffsets )
{
    calculateClasOutputBufferSizeAndOffsets_kernel<<<clusterCount, 1, 0, stream>>>(
        clusterCount, d_templateSizes, d_clasAddressOffsets, reinterpret_cast<unsigned long long*>( d_outputSizeInBytes ) );
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


void makeTemplatesArgsDataForGetSizes( CUstream&                                 stream,
                                       const CUdeviceptr*                        d_templateAddresses,
                                       const uint32_t                            numTemplates,
                                       OptixClusterAccelBuildInputTemplatesArgs* d_templatesArgs )
{
    dim3 blockSize( 32, 1, 1 );
    dim3 gridSize( ( numTemplates + blockSize.x - 1 ) / blockSize.x, 1, 1 );
    makeTemplatesArgsDataForGetSizes_kernel<<<gridSize, blockSize, 0, stream>>>( d_templateAddresses, numTemplates, d_templatesArgs );
}


__global__ void makeTemplatesArgsData_kernel( const Cluster*                            d_clusters,
                                              const uint32_t                            clusterCount,
                                              const CUdeviceptr*                        d_templateAddresses,
                                              OptixClusterAccelBuildInputTemplatesArgs* d_templatesArgs )
{
    const uint32_t clusterId = blockIdx.x * blockDim.x + threadIdx.x;
    if( clusterId >= clusterCount )
        return;

    const Cluster& cluster = d_clusters[clusterId];

    OptixClusterAccelBuildInputTemplatesArgs args = { 0 };
    args.clusterTemplate                          = d_templateAddresses[clusterId];
    args.vertexBuffer                             = reinterpret_cast<CUdeviceptr>( cluster.d_positions );
    args.vertexStrideInBytes                      = cluster.vertexStrideInBytes;

    d_templatesArgs[clusterId] = args;
}


void makeTemplatesArgsData( CUstream&                                 stream,
                            const Cluster*                            d_clusters,
                            const uint32_t                            clusterCount,
                            const CUdeviceptr*                        d_templateAddresses,
                            OptixClusterAccelBuildInputTemplatesArgs* d_templatesArgs )
{
    dim3 blockSize( 32, 1, 1 );
    dim3 gridSize( ( clusterCount + blockSize.x - 1 ) / blockSize.x, 1, 1 );
    makeTemplatesArgsData_kernel<<<gridSize, blockSize, 0, stream>>>( d_clusters, clusterCount, d_templateAddresses, d_templatesArgs );
}


__global__ void makeInputTrianglesArgsData_kernel( const Cluster*                            d_clusters,
                                                   const uint32_t                            clusterCount,
                                                   OptixClusterAccelBuildInputTrianglesArgs* d_trianglesArgs )
{
    const uint32_t clusterId = blockIdx.x * blockDim.x + threadIdx.x;
    if( clusterId >= clusterCount )
        return;

    const Cluster& cluster = d_clusters[clusterId];

    OptixClusterAccelBuildInputTrianglesArgs args = { 0 };
    args.clusterId                                = clusterId;  // global cluster ID for shading
    args.clusterFlags                             = OPTIX_CLUSTER_ACCEL_CLUSTER_FLAG_NONE;
    args.basePrimitiveInfo.primitiveFlags         = (unsigned int)OPTIX_CLUSTER_ACCEL_PRIMITIVE_FLAG_DISABLE_ANYHIT;
    args.triangleCount                            = cluster.triangleCount;
    args.vertexCount                              = cluster.vertexCount;
    args.indexFormat                              = cluster.indexFormat;
    args.indexBufferStrideInBytes                 = cluster.indexBufferStrideInBytes;
    args.vertexBufferStrideInBytes                = cluster.vertexStrideInBytes;
    args.indexBuffer                              = reinterpret_cast<CUdeviceptr>( cluster.d_indices );
    args.vertexBuffer                             = reinterpret_cast<CUdeviceptr>( cluster.d_positions );

    d_trianglesArgs[clusterId] = args;
}


void makeInputTrianglesArgsData( CUstream&                                 stream,
                                 const Cluster*                            d_clusters,
                                 const uint32_t                            clusterCount,
                                 OptixClusterAccelBuildInputTrianglesArgs* d_trianglesArgs )
{
    dim3 blockSize( 32, 1, 1 );
    dim3 gridSize( ( clusterCount + blockSize.x - 1 ) / blockSize.x, 1, 1 );
    makeInputTrianglesArgsData_kernel<<<gridSize, blockSize, 0, stream>>>( d_clusters, clusterCount, d_trianglesArgs );
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
    args.clusterHandlesCount = static_cast<uint32_t>( d_clusterOffsets[instanceId + 1] - d_clusterOffsets[instanceId] );
    args.clusterHandlesBuffer = CUdeviceptr( d_clasPtrs + d_clusterOffsets[instanceId] );

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


__global__ void copyGasHandlesToInstanceArray_kernel( OptixInstance* d_instances, const CUdeviceptr* d_gasHandles, const uint32_t instanceCount )
{
    const uint32_t instanceId = blockIdx.x * blockDim.x + threadIdx.x;
    if( instanceId >= instanceCount )
        return;

    d_instances[instanceId].traversableHandle = static_cast<OptixTraversableHandle>( d_gasHandles[instanceId] );
}


void copyGasHandlesToInstanceArray( CUstream& stream, OptixInstance* d_instances, const CUdeviceptr* d_gasHandles, const uint32_t instanceCount )
{
    dim3 blockSize( 32, 1, 1 );
    dim3 gridSize( ( instanceCount + blockSize.x - 1 ) / blockSize.x, 1, 1 );
    copyGasHandlesToInstanceArray_kernel<<<gridSize, blockSize, 0, stream>>>( d_instances, d_gasHandles, instanceCount );
}