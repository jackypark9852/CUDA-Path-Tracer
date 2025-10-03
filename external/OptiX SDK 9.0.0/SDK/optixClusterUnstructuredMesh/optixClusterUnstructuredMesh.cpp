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

#include <assert.h>
#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sampleConfig.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Scene.h>
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>

#include "optixClusterUnstructuredMesh.h"
#include "unstructuredClusterUtils.h"

#include <array>
#include <cstring>
#include <fstream>
#include <imgui/imgui.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>


bool g_resizeDirty = false;
bool g_minimized   = false;

// Camera state
bool             g_cameraChanged = true;
sutil::Camera    g_camera;
sutil::Trackball g_trackball;

// Mouse state
int32_t g_mouseButton = -1;

// device supported max value, query from DeviceProperty
uint32_t g_maxTrianglesPerCluster = 0;
uint32_t g_maxVerticesPerCluster  = 0;


//------------------------------------------------------------------------------
//
// Local types
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;


// The main goal of this sample is demonstrating how to use the cluster API on pre-clustered unstructured mesh.
// The cluster's vertices are deformed every frame but the topology information keeps the same.
// In this sample, we create an unique template for each cluster and reuse the templates every frame.

struct UnstructuredClusterState
{
    OptixDeviceContext                                 context = 0;
    CUstream                                           stream  = 0;

    sutil::Scene                                       scene              = {};
    size_t                                             totalTriangleCount = 0;

    Cluster*                                           d_clusters       = nullptr;
    uint32_t                                           clusterCount     = 0;
    size_t*                                            d_clusterOffsets = nullptr;

    CUdeviceptr*                                       d_clasPtrsBuffer     = nullptr;  // address of each CLAS in the cluster buffer
    CUdeviceptr                                        d_clasBuffer         = 0;
    size_t                                             clasBufferSize       = 0;

    CUdeviceptr                                        d_gasBuffer     = 0;
    size_t                                             gasBufferSize   = 0;
    CUdeviceptr*                                       d_gasPtrsBuffer = nullptr;       // handles in device memory

    uint32_t                                           maxClustersPerInstance = 0;
    uint32_t                                           maxTrianglesPerCluster = 0;
    uint32_t                                           maxVerticesPerCluster  = 0;

    // Buffers used to create templates. They are created once but need to be persistent throughout the app's run time.
    CUdeviceptr                                        d_templateBuffer     = 0;        // template buffer
    CUdeviceptr*                                       d_templatePtrsBuffer = nullptr;  // address of each template in the template buffer
    uint32_t*                                          d_templateSizeData   = nullptr;  // template size buffer

    //avoid re-alloc every frame
    CUdeviceptr                                        d_tempBuffer           = 0;        // temp buffer in both create and instantiate templates and GAS
    size_t                                             tempBufferSize         = 0; 
    size_t*                                            d_clasAddressOffsets   = nullptr;  // CLAS address offsets related to the big CLAS buffer
    OptixClusterAccelBuildInputTemplatesArgs*          d_clasTemplatesArgData = nullptr;  // used in buildCLASesFromTemplates()
    OptixClusterAccelBuildInputClustersArgs*           d_clustersArgs         = nullptr;

    OptixInstance*                                     d_instances         = nullptr;
    OptixBuildInput                                    iasInstanceInput    = {};
    OptixTraversableHandle                             iasHandle           = 0;
    CUdeviceptr                                        d_iasTempBuffer     = 0;
    size_t                                             iasTempBufferSize   = 0;
    CUdeviceptr                                        d_iasOutputBuffer   = 0;
    size_t                                             iasOutputBufferSize = 0;

    OptixModule                                        ptxModule              = 0;
    OptixPipelineCompileOptions                        pipelineCompileOptions = {};
    OptixPipeline                                      pipeline               = 0;

    OptixProgramGroup                                  raygenProgGroup = 0;
    OptixProgramGroup                                  missGroup       = 0;
    OptixProgramGroup                                  hitGroup        = 0;
    OptixShaderBindingTable                            sbt             = {};

    Params                                             params   = {};
    Params*                                            d_params = 0;

    float                                              time = 0.f;
}; //clang format off


//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------
static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        g_mouseButton = button;
        g_trackball.startTracking( static_cast<int>( xpos ), static_cast<int>( ypos ) );
    }
    else
    {
        g_mouseButton = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    Params* params = static_cast<Params*>( glfwGetWindowUserPointer( window ) );

    if( g_mouseButton == GLFW_MOUSE_BUTTON_LEFT )
    {
        g_trackball.setViewMode( sutil::Trackball::LookAtFixed );
        g_trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        g_cameraChanged = true;
    }
    else if( g_mouseButton == GLFW_MOUSE_BUTTON_RIGHT )
    {
        g_trackball.setViewMode( sutil::Trackball::EyeFixed );
        g_trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), params->width, params->height );
        g_cameraChanged = true;
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( g_minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    Params* params  = static_cast<Params*>( glfwGetWindowUserPointer( window ) );
    params->width   = res_x;
    params->height  = res_y;
    g_cameraChanged = true;
    g_resizeDirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    g_minimized = ( iconified > 0 );
}


static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/ )
{
    if( action == GLFW_PRESS )
    {
        if( key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE )
        {
            glfwSetWindowShouldClose( window, true );
        }
    }
    else if( key == GLFW_KEY_G )
    {
        // toggle UI draw
    }
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( g_trackball.wheelEvent( static_cast<int>( yscroll ) ) )
        g_cameraChanged = true;
}


//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>           File for image output\n";
    std::cerr << "         --model | -m <model.gltf>        Specify pre-clustered model to render (non-clustered file will fail)\n";
    std::cerr << "         --no-gl-interop                  Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>           Set image dimensions; defaults to 1024x768\n";
    std::cerr << "         --help | -h                      Print this usage message\n";
    std::cerr << "\n";
    exit( 0 );
}


void initLaunchParams( UnstructuredClusterState& state )
{
    state.params.frameBuffer   = nullptr;  // Will be set when output buffer is mapped
    state.params.subframeIndex = 0u;

    CUDA_CHECK( cudaStreamCreate( &state.stream ) );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_params ), sizeof( Params ) ) );
}


void handleCameraUpdate( Params& params )
{
    if( !g_cameraChanged )
        return;
    g_cameraChanged = false;

    g_camera.setAspectRatio( static_cast<float>( params.width ) / static_cast<float>( params.height ) );
    params.eye = g_camera.eye();
    g_camera.UVWFrame( params.U, params.V, params.W );
}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& outputBuffer, Params& params )
{
    if( !g_resizeDirty )
        return;
    g_resizeDirty = false;

    outputBuffer.resize( params.width, params.height );
}


void updateState( sutil::CUDAOutputBuffer<uchar4>& outputBuffer, Params& params )
{
    handleCameraUpdate( params );
    handleResize( outputBuffer, params );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& outputBuffer, UnstructuredClusterState& state )
{
    // Launch
    uchar4* resultBufferData = outputBuffer.map();
    state.params.frameBuffer = resultBufferData;
    CUDA_CHECK( cudaMemcpyAsync(
        reinterpret_cast<void*>( state.d_params ),
        &state.params,
        sizeof( Params ),
        cudaMemcpyHostToDevice,
        state.stream
    ) );

    // clang format off
    OPTIX_CHECK( optixLaunch( 
        state.pipeline, 
        state.stream, 
        reinterpret_cast<CUdeviceptr>( state.d_params ),
        sizeof( Params ), 
        &state.sbt,
        state.params.width,   // launch width
        state.params.height,  // launch height
        1                     // launch depth
    ) );

    outputBuffer.unmap();
}


void displaySubframe( sutil::CUDAOutputBuffer<uchar4>& outputBuffer, sutil::GLDisplay& glDisplay, GLFWwindow* window )
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    glDisplay.display( outputBuffer.width(), outputBuffer.height(), framebuf_res_x, framebuf_res_y, outputBuffer.getPBO() );
}


static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */ )
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: " << message << "\n";
}


void initCameraState( UnstructuredClusterState& state )
{
    sutil::Aabb scene_aabb;
    scene_aabb.invalidate();
    for( const auto& instance : state.scene.instances() )
        scene_aabb.include( instance->world_aabb );

    g_camera.setLookat( scene_aabb.center() );
    g_camera.setEye( scene_aabb.center() + make_float3( 0.0f, 0.0f, 2.3f ) * scene_aabb.maxExtent() );
    g_camera.setFovY( 35.0f );
    g_cameraChanged = true;

    g_trackball.setCamera( &g_camera );
    g_trackball.setMoveSpeed( 10.0f );
    g_trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ),
                                   make_float3( 0.0f, 1.0f, 0.0f ) );
    g_trackball.setGimbalLock( true );
}


void createContext( UnstructuredClusterState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    options.validationMode            = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &state.context ) );

    int isClustersSupported = 0;
    OPTIX_CHECK( optixDeviceContextGetProperty( state.context, OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL,
                                                &isClustersSupported, sizeof( isClustersSupported ) ) );
    if( !isClustersSupported )
    {
        std::cerr << "Cluster API is not supported on this GPU" << std::endl;
        exit( 1 );
    }

    OPTIX_CHECK( optixDeviceContextGetProperty( state.context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_TRIANGLES,
                                                &g_maxTrianglesPerCluster, sizeof( g_maxTrianglesPerCluster ) ) );

    OPTIX_CHECK( optixDeviceContextGetProperty( state.context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_VERTICES,
                                                &g_maxVerticesPerCluster, sizeof( g_maxVerticesPerCluster ) ) );
}

//------------------------------------------------------------------------------
//
// Helper functions for creating templates, CLAS etc
//
//------------------------------------------------------------------------------
void inline resizeDeviceBuffer( CUdeviceptr& inOuputBuffer, size_t& oldSize, size_t newSize, CUstream stream = 0 )
{
    if( oldSize < newSize )
    {
        CUdeviceptr tmpBuffer = 0;
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &tmpBuffer ), newSize, stream ) );
        CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( tmpBuffer ), reinterpret_cast<void*>( inOuputBuffer ),
                                     oldSize, cudaMemcpyDeviceToDevice, stream ) );
        std::swap( inOuputBuffer, tmpBuffer );
        std::swap( oldSize, newSize );
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( tmpBuffer ), stream ) );
    }
}


void inline initUnstructuredClusterTemplates( UnstructuredClusterState& state )
{
    uint32_t maxNumTemplates = state.clusterCount;

    OptixClusterAccelBuildInput buildInputs            = {};
    buildInputs.type                                   = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_TRIANGLES;
    buildInputs.triangles.flags                        = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
    buildInputs.triangles.maxArgCount                  = maxNumTemplates;
    buildInputs.triangles.vertexFormat                 = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInputs.triangles.maxUniqueSbtIndexCountPerArg = 1;
    buildInputs.triangles.maxTriangleCountPerArg       = state.maxTrianglesPerCluster;
    buildInputs.triangles.maxVertexCountPerArg         = state.maxVerticesPerCluster;

    OptixAccelBufferSizes templateBufferSizes = {};
    OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( state.context, OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
                                                      &buildInputs, &templateBufferSizes ) );

    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_templateBuffer ),
                                 templateBufferSizes.outputSizeInBytes, state.stream ) );
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_templatePtrsBuffer ),
                                 maxNumTemplates * sizeof( CUdeviceptr ), state.stream ) );
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_templateSizeData ),
                                 maxNumTemplates * sizeof( uint32_t ), state.stream ) );

    resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, templateBufferSizes.tempSizeInBytes, state.stream );
    CUDA_CHECK( cudaMemsetAsync( reinterpret_cast<void*>( state.d_tempBuffer ), 0, state.tempBufferSize, state.stream ) );

    OptixClusterAccelBuildInputTrianglesArgs* d_inputTrianglesArgs = nullptr;
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_inputTrianglesArgs ),
                                 maxNumTemplates * sizeof( OptixClusterAccelBuildInputTrianglesArgs ), state.stream ) );
    makeInputTrianglesArgsData( state.stream, state.d_clusters, maxNumTemplates, d_inputTrianglesArgs );

    OptixClusterAccelBuildModeDesc clusterAccelBuildModeDec          = {};
    clusterAccelBuildModeDec.mode                                    = OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
    clusterAccelBuildModeDec.implicitDest.outputBuffer               = state.d_templateBuffer;
    clusterAccelBuildModeDec.implicitDest.outputBufferSizeInBytes    = templateBufferSizes.outputSizeInBytes;
    clusterAccelBuildModeDec.implicitDest.tempBuffer                 = state.d_tempBuffer;
    clusterAccelBuildModeDec.implicitDest.tempBufferSizeInBytes      = state.tempBufferSize;
    clusterAccelBuildModeDec.implicitDest.outputHandlesBuffer        = reinterpret_cast<CUdeviceptr>( state.d_templatePtrsBuffer );
    clusterAccelBuildModeDec.implicitDest.outputHandlesStrideInBytes = sizeof( CUdeviceptr );

    OPTIX_CHECK( optixClusterAccelBuild( state.context, state.stream, &clusterAccelBuildModeDec, &buildInputs,
                                         reinterpret_cast<CUdeviceptr>( d_inputTrianglesArgs ), 0 /*argsCount*/, 0 ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( d_inputTrianglesArgs ), state.stream ) );

    /*------------ Compute Templates Sizes -------------*/
    // Instantiate templates without providing the vertex positions returns the worst case size for each grid/cluster size
    OptixClusterAccelBuildInputTemplatesArgs* d_templatesArgs = nullptr;
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_templatesArgs ),
                                 maxNumTemplates * sizeof( OptixClusterAccelBuildInputTemplatesArgs ), state.stream ) );
    makeTemplatesArgsDataForGetSizes( state.stream, state.d_templatePtrsBuffer, maxNumTemplates, d_templatesArgs );

    buildInputs.type = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES;

    // it is the recommended practice to estimate memory usage again before the GET_SIZES build
    OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( state.context, OPTIX_CLUSTER_ACCEL_BUILD_MODE_GET_SIZES,
                                                      &buildInputs, &templateBufferSizes ) );

    resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, templateBufferSizes.tempSizeInBytes, state.stream );
    CUDA_CHECK( cudaMemsetAsync( reinterpret_cast<void*>( state.d_tempBuffer ), 0, state.tempBufferSize, state.stream ) );

    OptixClusterAccelBuildModeDesc getTemplateSizesBuildDesc   = {};
    getTemplateSizesBuildDesc.mode                             = OPTIX_CLUSTER_ACCEL_BUILD_MODE_GET_SIZES;
    getTemplateSizesBuildDesc.getSize.tempBuffer               = state.d_tempBuffer;
    getTemplateSizesBuildDesc.getSize.tempBufferSizeInBytes    = state.tempBufferSize;
    getTemplateSizesBuildDesc.getSize.outputSizesBuffer        = reinterpret_cast<CUdeviceptr>( state.d_templateSizeData );
    getTemplateSizesBuildDesc.getSize.outputSizesStrideInBytes = sizeof( uint32_t );

    OPTIX_CHECK( optixClusterAccelBuild( state.context, state.stream, &getTemplateSizesBuildDesc, &buildInputs,
                                         reinterpret_cast<CUdeviceptr>( d_templatesArgs ), 0 /*argsCount*/, 0 ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( d_templatesArgs ), state.stream ) );
}

void inline getClasOutputBufferSizeAndOffsets( UnstructuredClusterState& state )
{
    resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, sizeof( size_t ), state.stream );
    CUDA_CHECK( cudaMemsetAsync( (void*)state.d_tempBuffer, 0, state.tempBufferSize, state.stream ) );
    size_t* d_outputSizeInBytes = reinterpret_cast<size_t*>( state.d_tempBuffer );
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clasAddressOffsets ),
                                 state.clusterCount * sizeof( size_t ), state.stream ) );
    calculateClasOutputBufferSizeAndOffsets( state.stream, state.clusterCount, state.d_templateSizeData,
                                             d_outputSizeInBytes, state.d_clasAddressOffsets );
    CUDA_CHECK( cudaMemcpyAsync( &state.clasBufferSize, d_outputSizeInBytes, sizeof( size_t ),
                                 cudaMemcpyDeviceToHost, state.stream ) );
}

void inline buildCLASesFromTemplates( UnstructuredClusterState& state )
{

    OptixClusterAccelBuildInput buildInputs            = {};
    buildInputs.type                                   = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES;
    buildInputs.triangles.flags                        = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
    buildInputs.triangles.maxArgCount                  = state.clusterCount;
    buildInputs.triangles.vertexFormat                 = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInputs.triangles.maxUniqueSbtIndexCountPerArg = 1;
    buildInputs.triangles.maxTriangleCountPerArg       = state.maxTrianglesPerCluster;
    buildInputs.triangles.maxVertexCountPerArg         = state.maxVerticesPerCluster;

    if( !state.d_clasBuffer )
    {
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clasBuffer ), state.clasBufferSize, state.stream ) );
    }

    if( !state.d_clasPtrsBuffer )
    {
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clasPtrsBuffer ),
                                     state.clusterCount * sizeof( CUdeviceptr ), state.stream ) );
        CUDA_CHECK( cudaMemsetAsync( (void*)state.d_clasPtrsBuffer, 0, state.clusterCount * sizeof( CUdeviceptr ), state.stream ) );
    }

    if( !state.d_clasTemplatesArgData )
    {
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clasTemplatesArgData ),
                                     state.clusterCount * sizeof( OptixClusterAccelBuildInputTemplatesArgs ), state.stream ) );
    }
    makeTemplatesArgsData( state.stream, state.d_clusters, state.clusterCount, state.d_templatePtrsBuffer, state.d_clasTemplatesArgData );

    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( state.context, OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
                                                      &buildInputs, &bufferSizes ) );

    resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, bufferSizes.tempSizeInBytes, state.stream );
    CUDA_CHECK( cudaMemsetAsync( reinterpret_cast<void*>( state.d_tempBuffer ), 0, state.tempBufferSize, state.stream ) );

    assignExplicitAddresses( state.stream, state.clusterCount, state.d_clasAddressOffsets, state.d_clasBuffer, state.d_clasPtrsBuffer );

    OptixClusterAccelBuildModeDesc clasBuildDesc          = {};
    clasBuildDesc.mode                                    = OPTIX_CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS;
    clasBuildDesc.explicitDest.destAddressesBuffer        = reinterpret_cast<CUdeviceptr>( state.d_clasPtrsBuffer );
    clasBuildDesc.explicitDest.destAddressesStrideInBytes = sizeof( CUdeviceptr );
    clasBuildDesc.explicitDest.tempBuffer                 = state.d_tempBuffer;
    clasBuildDesc.explicitDest.tempBufferSizeInBytes      = state.tempBufferSize;
    clasBuildDesc.explicitDest.outputHandlesBuffer        = reinterpret_cast<CUdeviceptr>( state.d_clasPtrsBuffer );
    clasBuildDesc.explicitDest.outputHandlesStrideInBytes = sizeof( CUdeviceptr );

    OPTIX_CHECK( optixClusterAccelBuild( state.context, state.stream, &clasBuildDesc, &buildInputs,
                                         reinterpret_cast<CUdeviceptr>( state.d_clasTemplatesArgData ), 0 /*argsCount*/, 0 ) );
}


void inline makeClusters( UnstructuredClusterState& state )
{
    using namespace sutil;
    const std::vector<std::shared_ptr<Scene::MeshGroup>>& meshes    = state.scene.meshes();
    const std::vector<std::shared_ptr<Scene::Instance>>&  instances = state.scene.instances();

    std::vector<size_t> clusterOffsets( instances.size() + 1, 0 );

    std::vector<Cluster> clusters;
    for( size_t i = 0; i < instances.size(); ++i )
    {
        std::shared_ptr<Scene::MeshGroup> mesh          = meshes[instances[i]->mesh_idx];
        const size_t                      num_subMeshes = mesh->indices.size();  // each submesh is a cluster

        clusterOffsets[i + 1]        = clusterOffsets[i] + num_subMeshes;
        state.maxClustersPerInstance = std::max( state.maxClustersPerInstance, static_cast<uint32_t>( num_subMeshes ) );

        for( size_t j = 0; j < num_subMeshes; ++j )
        {
            Cluster cluster;
            cluster.triangleCount            = mesh->indices[j].count / 3;
            cluster.vertexCount              = mesh->positions[j].count;
            cluster.vertexStrideInBytes      = mesh->positions[j].byte_stride;
            cluster.d_indices                = mesh->indices[j].data;
            cluster.indexFormat              = static_cast<OptixClusterAccelIndicesFormat>( mesh->indices[j].elmt_byte_size );
            cluster.indexBufferStrideInBytes = mesh->indices[j].byte_stride;
            cluster.d_positions              = mesh->positions[j].data;
            clusters.push_back( cluster );

            state.totalTriangleCount += cluster.triangleCount;

            state.maxTrianglesPerCluster = std::max( state.maxTrianglesPerCluster, cluster.triangleCount );
            state.maxVerticesPerCluster  = std::max( state.maxVerticesPerCluster, cluster.vertexCount );
        }
    }

    if( state.maxTrianglesPerCluster > g_maxTrianglesPerCluster )
    {
        std::cerr << "Max triangles within a cluster is greater than the device-supported max value" << std::endl;
        exit( 1 );
    }

    if( state.maxVerticesPerCluster > g_maxVerticesPerCluster )
    {
        std::cerr << "Max vertices within a cluster is greater than the device-supported max value" << std::endl;
        exit( 1 );
    }

    state.clusterCount = static_cast<uint32_t>( clusters.size() );
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clusters ), clusters.size() * sizeof( Cluster ),
                                 state.stream ) );
    CUDA_CHECK( cudaMemcpyAsync( state.d_clusters, clusters.data(), clusters.size() * sizeof( Cluster ),
                                 cudaMemcpyHostToDevice, state.stream ) );

    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clusterOffsets ),
                                 clusterOffsets.size() * sizeof( size_t ), state.stream ) );
    CUDA_CHECK( cudaMemcpyAsync( state.d_clusterOffsets, clusterOffsets.data(),
                                 clusterOffsets.size() * sizeof( size_t ), cudaMemcpyHostToDevice, state.stream ) );

    printf( "meshCount: %zu, instanceCount: %zu, clusterCount: %d, triangleCount: %zu\n", meshes.size(),
            instances.size(), state.clusterCount, state.totalTriangleCount );
}


void buildGAS( UnstructuredClusterState& state )
{
    deformClusterVertices( state.stream, state.clusterCount, state.time, state.d_clusters );

    buildCLASesFromTemplates( state );

    // Allocate and build GASes from CLASes
    {
        const std::vector<std::shared_ptr<sutil::Scene::Instance>>& instances     = state.scene.instances();
        const uint32_t                                              instanceCount = (uint32_t)instances.size();

        const uint32_t maxClusterCount       = state.clusterCount;
        const uint32_t maxClusterCountPerGas = state.maxClustersPerInstance;

        OptixClusterAccelBuildInput buildInputs    = {};
        buildInputs.type                           = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS;
        buildInputs.clusters.flags                 = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
        buildInputs.clusters.maxArgCount           = instanceCount;
        buildInputs.clusters.maxTotalClusterCount  = maxClusterCount;
        buildInputs.clusters.maxClusterCountPerArg = maxClusterCountPerGas;

        OptixAccelBufferSizes gasBufferSizes{};
        OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( state.context, OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
                                                          &buildInputs, &gasBufferSizes ) );

        resizeDeviceBuffer( state.d_gasBuffer, state.gasBufferSize, gasBufferSizes.outputSizeInBytes, state.stream );
        if( !state.d_gasPtrsBuffer )
        {
            CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_gasPtrsBuffer ),
                                         instanceCount * sizeof( CUdeviceptr ), state.stream ) );
        }

        resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, gasBufferSizes.tempSizeInBytes, state.stream );

        if( !state.d_clustersArgs )
        {
            CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clustersArgs ),
                                         instanceCount * sizeof( OptixClusterAccelBuildInputClustersArgs ), state.stream ) );
        }
        makeClustersArgsData( state.stream, state.d_clusterOffsets, instanceCount, state.d_clasPtrsBuffer, state.d_clustersArgs );

        OptixClusterAccelBuildModeDesc gasBuildDesc          = {};
        gasBuildDesc.mode                                    = OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
        gasBuildDesc.implicitDest.outputBuffer               = state.d_gasBuffer;
        gasBuildDesc.implicitDest.outputBufferSizeInBytes    = state.gasBufferSize;
        gasBuildDesc.implicitDest.tempBuffer                 = state.d_tempBuffer;
        gasBuildDesc.implicitDest.tempBufferSizeInBytes      = state.tempBufferSize;
        gasBuildDesc.implicitDest.outputHandlesBuffer        = reinterpret_cast<CUdeviceptr>( state.d_gasPtrsBuffer );
        gasBuildDesc.implicitDest.outputHandlesStrideInBytes = sizeof( CUdeviceptr );

        OPTIX_CHECK( optixClusterAccelBuild( state.context, state.stream, &gasBuildDesc, &buildInputs,
                                             reinterpret_cast<CUdeviceptr>( state.d_clustersArgs ), 0 /*argsCount*/, 0 ) );
    }
}

void buildIAS( UnstructuredClusterState& state )
{
    // Build the IAS
    const std::vector<std::shared_ptr<sutil::Scene::Instance>>& meshInstances = state.scene.instances();
    assert( meshInstances.size() == 1 );

    if( !state.d_instances )
    {
        std::vector<OptixInstance> instances( meshInstances.size() );
        for( size_t i = 0; i < instances.size(); ++i )
        {
            memcpy( instances[i].transform, meshInstances[i]->transform.getData(), sizeof( float ) * 12 );
            instances[i].sbtOffset      = static_cast<unsigned int>( i );
            instances[i].visibilityMask = 255;
        }
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_instances ),
                                     instances.size() * sizeof( OptixInstance ), state.stream ) );
        CUDA_CHECK( cudaMemcpyAsync( state.d_instances, instances.data(), instances.size() * sizeof( OptixInstance ),
                                     cudaMemcpyHostToDevice, state.stream ) );
    }

    copyGasHandlesToInstanceArray( state.stream, state.d_instances, state.d_gasPtrsBuffer,
                                   static_cast<uint32_t>( meshInstances.size() ) );

    state.iasInstanceInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    state.iasInstanceInput.instanceArray.instances    = reinterpret_cast<CUdeviceptr>( state.d_instances );
    state.iasInstanceInput.instanceArray.numInstances = static_cast<int>( meshInstances.size() );

    OptixAccelBuildOptions iasAccelOptions = {};
    iasAccelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    iasAccelOptions.motionOptions.numKeys  = 1;
    iasAccelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &iasAccelOptions, &state.iasInstanceInput, 1, &iasBufferSizes ) );

    resizeDeviceBuffer( state.d_iasTempBuffer, state.iasTempBufferSize, iasBufferSizes.tempSizeInBytes, state.stream );

    resizeDeviceBuffer( state.d_iasOutputBuffer, state.iasOutputBufferSize, iasBufferSizes.outputSizeInBytes, state.stream );

    OPTIX_CHECK( optixAccelBuild( 
        state.context,
        state.stream,
        &iasAccelOptions,
        &state.iasInstanceInput,
        1,
        state.d_iasTempBuffer,
        state.iasTempBufferSize,
        state.d_iasOutputBuffer,
        state.iasOutputBufferSize,
        &state.iasHandle,
        NULL,
        0 
    ) );  // clang format off

    state.params.handle = state.iasHandle;
}

void createModule( UnstructuredClusterState& state )
{
    OptixModuleCompileOptions moduleCompileOptions = {};
#if OPTIX_DEBUG_DEVICE_CODE
    moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#endif

    state.pipelineCompileOptions.usesMotionBlur        = false;
    state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    state.pipelineCompileOptions.numPayloadValues      = 3;
    state.pipelineCompileOptions.numAttributeValues    = 2;
    state.pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    state.pipelineCompileOptions.allowClusteredGeometry           = true;

    size_t inputSize = 0;
    const char* input = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixClusterUnstructuredMesh.cu", inputSize );

    OPTIX_CHECK_LOG( optixModuleCreate(
        state.context,
        &moduleCompileOptions,
        &state.pipelineCompileOptions,
        input,
        inputSize,
        LOG,
        &LOG_SIZE,
        &state.ptxModule
    ) );
}


void createProgramGroups( UnstructuredClusterState& state )
{
    OptixProgramGroupOptions programGroupOptions = {};

    {
        OptixProgramGroupDesc raygenProgramGroupDesc    = {};
        raygenProgramGroupDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenProgramGroupDesc.raygen.module            = state.ptxModule;
        raygenProgramGroupDesc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &raygenProgramGroupDesc,
            1,  // num program groups
            &programGroupOptions,
            LOG,
            &LOG_SIZE,
            &state.raygenProgGroup
        ) );
    }

    {
        OptixProgramGroupDesc missProgramGroupDesc  = {};
        missProgramGroupDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missProgramGroupDesc.miss.module            = state.ptxModule;
        missProgramGroupDesc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &missProgramGroupDesc,
            1,  // num program groups
            &programGroupOptions,
            LOG,
            &LOG_SIZE,
            &state.missGroup
        ) );
    }

    {
        OptixProgramGroupDesc hitProgramGroupDesc        = {};
        hitProgramGroupDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitProgramGroupDesc.hitgroup.moduleCH            = state.ptxModule;
        hitProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            state.context,
            &hitProgramGroupDesc,
            1,  // num program groups
            &programGroupOptions,
            LOG,
            &LOG_SIZE,
            &state.hitGroup
        ) );
    }
}


void createPipeline( UnstructuredClusterState& state )
{
    OptixProgramGroup programGroups[] = 
    { 
        state.raygenProgGroup,
        state.missGroup,
        state.hitGroup
    };

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth            = 1;

    OPTIX_CHECK_LOG( optixPipelineCreate(
        state.context,
        &state.pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups,
        sizeof( programGroups ) / sizeof( programGroups[0] ),
        LOG,
        &LOG_SIZE,
        &state.pipeline
    ) );  //clang format off

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stackSizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.raygenProgGroup, &stackSizes, state.pipeline ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.missGroup, &stackSizes, state.pipeline ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes( state.hitGroup, &stackSizes, state.pipeline ) );

    uint32_t maxTraceDepth = 1;
    uint32_t maxCCDepth    = 0;
    uint32_t maxDCDepth    = 0;
    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;
    OPTIX_CHECK( optixUtilComputeStackSizes( 
        &stackSizes, 
        maxTraceDepth, 
        maxCCDepth, 
        maxDCDepth, 
        &directCallableStackSizeFromTraversal,
        &directCallableStackSizeFromState, 
        &continuationStackSize 
    ) );//clang format off

    // This is 2 since the largest depth is IAS->GAS
    const uint32_t max_traversable_graph_depth = 2;

    OPTIX_CHECK( optixPipelineSetStackSize( 
        state.pipeline, 
        directCallableStackSizeFromTraversal, 
        directCallableStackSizeFromState,
        continuationStackSize, 
        max_traversable_graph_depth 
    ) );//clang format off
}


void createSBT( UnstructuredClusterState& state )
{
    CUdeviceptr  d_raygenRecord;
    const size_t raygenRecordSize = sizeof( RayGenRecord );
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_raygenRecord ), raygenRecordSize, state.stream  ) );

    RayGenRecord rgSBT = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygenProgGroup, &rgSBT ) );

    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_raygenRecord ), &rgSBT, raygenRecordSize,
                                 cudaMemcpyHostToDevice, state.stream ) );

    CUdeviceptr  d_missRecords;
    const size_t missRecordSize = sizeof( MissRecord );
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_missRecords ), missRecordSize, state.stream  ) );

    MissRecord missSBT[1];
    OPTIX_CHECK( optixSbtRecordPackHeader( state.missGroup, &missSBT[0] ) );
    missSBT[0].data.bg_color = make_float4( 0.0f );

    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_missRecords ), missSBT, missRecordSize,
                                 cudaMemcpyHostToDevice, state.stream ) );

    CUdeviceptr                 d_hitgroupRecords;
    const size_t                hitgroupRecordSize = sizeof( HitGroupRecord );
    std::vector<HitGroupRecord> hitgroupRecords;

    const std::vector<std::shared_ptr<sutil::Scene::Instance>>& instances = state.scene.instances();
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_hitgroupRecords ),
                                 hitgroupRecordSize * instances.size(), state.stream ) );
    hitgroupRecords.resize( instances.size() );

    for( int i = 0; i < static_cast<int>( hitgroupRecords.size() ); ++i )
    {
        const int sbtIdx = i;
        OPTIX_CHECK( optixSbtRecordPackHeader( state.hitGroup, &hitgroupRecords[sbtIdx] ) );
    }

    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_hitgroupRecords ), hitgroupRecords.data(),
                            hitgroupRecordSize * hitgroupRecords.size(), cudaMemcpyHostToDevice, state.stream ) );

    state.sbt.raygenRecord                = d_raygenRecord;
    state.sbt.missRecordBase              = d_missRecords;
    state.sbt.missRecordStrideInBytes     = static_cast<uint32_t>( missRecordSize );
    state.sbt.missRecordCount             = 1;
    state.sbt.hitgroupRecordBase          = d_hitgroupRecords;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>( hitgroupRecordSize );
    state.sbt.hitgroupRecordCount         = static_cast<uint32_t>( hitgroupRecords.size() );
}


void cleanupState( UnstructuredClusterState& state )
{
    state.scene.cleanup();
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clusters ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clusterOffsets ), state.stream ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clasPtrsBuffer ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clasBuffer ), state.stream ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_gasBuffer ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_gasPtrsBuffer ), state.stream ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_templateBuffer ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_templatePtrsBuffer ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_templateSizeData ), state.stream ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_tempBuffer ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clasAddressOffsets ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clasTemplatesArgData ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clustersArgs ), state.stream ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_instances ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_iasTempBuffer ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_iasOutputBuffer ), state.stream ) );

    OPTIX_CHECK( optixPipelineDestroy( state.pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.raygenProgGroup ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.missGroup ) );
    OPTIX_CHECK( optixProgramGroupDestroy( state.hitGroup ) );
    OPTIX_CHECK( optixModuleDestroy( state.ptxModule ) );
    OPTIX_CHECK( optixDeviceContextDestroy( state.context ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.sbt.raygenRecord ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.sbt.missRecordBase ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.sbt.hitgroupRecordBase ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_params ), state.stream ) );

    CUDA_CHECK( cudaStreamDestroy( state.stream ) );
}


void displayStats( UnstructuredClusterState& state, float& accelBuildTime, float& renderTime )
{
    constexpr std::chrono::duration<double> displayUpdateMinIntervalTime( 0.5 );
    static int32_t                          totalSubframeCount = 0;
    static int32_t                          lastUpdateFrames   = 0;
    static auto                             lastUpdateTime     = std::chrono::steady_clock::now();
    static char                             displayText[128];

    const auto curTime = std::chrono::steady_clock::now();

    sutil::beginFrameImGui();
    lastUpdateFrames++;

    if( curTime - lastUpdateTime > displayUpdateMinIntervalTime || totalSubframeCount == 0 )
    {
        sprintf( displayText,
                 "%5.1f fps\n\n"
                 "accel build : %8.2f ms\n"
                 "render      : %8.2f ms\n",
                 lastUpdateFrames / std::chrono::duration<double>( curTime - lastUpdateTime ).count(),
                 accelBuildTime / lastUpdateFrames, 
                 renderTime / lastUpdateFrames 
                );

        lastUpdateTime   = curTime;
        lastUpdateFrames = 0;
        accelBuildTime = renderTime = 0.f;
    }
    sutil::displayText( displayText, 10.0f, 10.0f );

    sutil::endFrameImGui();
    ++totalSubframeCount;
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    UnstructuredClusterState state;
    state.params.width                             = 1024;
    state.params.height                            = 768;
    state.time                                     = 0.f;
    sutil::CUDAOutputBufferType outputBufferType = sutil::CUDAOutputBufferType::GL_INTEROP;

    //
    // Parse command line options
    //
    std::string infile;
    std::string outfile;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg = argv[i];
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--no-gl-interop" )
        {
            outputBufferType = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if( ( arg == "-m" || arg == "--model" ) && i + 1 < argc )
        {
            infile = argv[++i];
        }
        else if( arg == "--file" || arg == "-f" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            outfile = argv[++i];
        }
        else if( arg.substr( 0, 6 ) == "--dim=" )
        {
            const std::string dims_arg = arg.substr( 6 );
            int               w, h;
            sutil::parseDimensions( dims_arg.c_str(), w, h );
            state.params.width  = w;
            state.params.height = h;
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        if ( infile.empty() )
            infile = sutil::sampleDataFilePath( "Duck/duck_clustered.gltf" );

        sutil::loadScene( infile.c_str(), state.scene );

        initCameraState( state );
        //
        // Set up OptiX state
        //
        createContext( state );

        createModule( state );
        createProgramGroups( state );
        createPipeline( state );
        createSBT( state );
        initLaunchParams( state );

        makeClusters( state );
        initUnstructuredClusterTemplates( state );
        getClasOutputBufferSizeAndOffsets( state );

        // create CUDA events for timing
        cudaEvent_t accelBuildTimeStart, accelBuildTimeStop, renderTimeStart, renderTimeStop;
        CUDA_CHECK( cudaEventCreate( &accelBuildTimeStart ) );
        CUDA_CHECK( cudaEventCreate( &accelBuildTimeStop ) );
        CUDA_CHECK( cudaEventCreate( &renderTimeStart ) );
        CUDA_CHECK( cudaEventCreate( &renderTimeStop ) );

        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixClusterUnstructuredMesh", state.params.width, state.params.height );
            glfwSetMouseButtonCallback( window, mouseButtonCallback );
            glfwSetCursorPosCallback( window, cursorPosCallback );
            glfwSetWindowSizeCallback( window, windowSizeCallback );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback( window, keyCallback );
            glfwSetScrollCallback( window, scrollCallback );
            glfwSetWindowUserPointer( window, &state.params );

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> outputBuffer( outputBufferType, state.params.width, state.params.height );

                outputBuffer.setStream( state.stream );
                sutil::GLDisplay glDisplay;

                float accelBuildTime( 0.f );
                float renderTime( 0.f );

                auto tstart = std::chrono::system_clock::now();

                do
                {
                    glfwPollEvents();

                    auto                          tnow = std::chrono::system_clock::now();
                    std::chrono::duration<double> time = tnow - tstart;
                    tstart                             = tnow;

                    state.time += (float)time.count();

                    CUDA_CHECK( cudaEventRecord( accelBuildTimeStart, state.stream ) );
                    buildGAS( state );
                    buildIAS( state );
                    CUDA_CHECK( cudaEventRecord( accelBuildTimeStop, state.stream ) );

                    updateState( outputBuffer, state.params );

                    CUDA_CHECK( cudaEventRecord( renderTimeStart, state.stream ) );
                    launchSubframe( outputBuffer, state );
                    CUDA_CHECK( cudaEventRecord( renderTimeStop, state.stream ) );

                    displaySubframe( outputBuffer, glDisplay, window );

                    CUDA_SYNC_CHECK();

                    float accelBuildDuration = 0.f;
                    CUDA_CHECK( cudaEventElapsedTime( &accelBuildDuration, accelBuildTimeStart, accelBuildTimeStop ) );
                    accelBuildTime += accelBuildDuration;

                    float renderBuildDuration = 0.f;
                    CUDA_CHECK( cudaEventElapsedTime( &renderBuildDuration, renderTimeStart, renderTimeStop ) );
                    renderTime += renderBuildDuration;

                    displayStats( state, accelBuildTime, renderTime );

                    glfwSwapBuffers( window );

                    ++state.params.subframeIndex;
                } while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );
        }
        else
        {
            if( outputBufferType == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            {
                // this scope is for outputBuffer, to ensure the destructor is called bfore glfwTerminate()

                sutil::CUDAOutputBuffer<uchar4> outputBuffer( outputBufferType, state.params.width, state.params.height );

                handleCameraUpdate( state.params );
                handleResize( outputBuffer, state.params );

                buildGAS( state );
                buildIAS( state );

                launchSubframe( outputBuffer, state );

                sutil::ImageBuffer buffer;
                buffer.data         = outputBuffer.getHostPointer();
                buffer.width        = outputBuffer.width();
                buffer.height       = outputBuffer.height();
                buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

                sutil::saveImage( outfile.c_str(), buffer, false );
            }

            if( outputBufferType == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

        cleanupState( state );

        CUDA_CHECK( cudaEventDestroy( accelBuildTimeStart ) );
        CUDA_CHECK( cudaEventDestroy( accelBuildTimeStop ) );
        CUDA_CHECK( cudaEventDestroy( renderTimeStart ) );
        CUDA_CHECK( cudaEventDestroy( renderTimeStop ) );
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
