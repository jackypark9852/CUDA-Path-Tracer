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

#include <glad/glad.h>  // Needs to be included before gl_interop

#include <assert.h>
#include <cuda_gl_interop.h>
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

#include "clusterUtils.h"
#include "optixClusterStructuredMesh.h"

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
float2  g_mousePos    = {};

float3 g_mousePosInWorld = {};


uint2 g_edgeSegments = { 8, 8 };
uint2 g_gridDims     = { 512, 512 };

AccelBuildMode g_accelBuildMode          = AccelBuildMode::CLUSTER;
bool           g_isAccelBuildModeChanged = false;
bool           g_isClusterInfoChanged     = true;

uint32_t g_maxClusterEdgeSegments = 0;


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


// The main goal of this sample is:
//  1) demonstrate how to use cluster triangle API with structured input mesh 
//  2) demonstrate the GAS build time difference between cluster triangle and flat triangle build
// the mesh is anminated with some random waves, thus the cluster and cluster GAS are being built every frames. 
// the flat triangle GAS is built every frames as well for comparison. 
// Cluster build achieves a dramatic performance improvement in terms of GAS build time (~ 14x with the default settings)
// when compares to the falt triangle build

struct StructuredClusterState
{
    OptixDeviceContext                                  context = 0;
    CUstream                                            stream = 0;

    sutil::Scene                                        scene              = {};
    size_t                                              totalTriangleCount = 0;

    // GAS
    std::unique_ptr<Accel>                              pAccel           = nullptr;
    ClusterTessConfig                                   tessConfig;
    Cluster*                                            d_clusters       = nullptr;
    uint32_t                                            clusterCount     = 0;
    size_t*                                             d_clusterOffsets = nullptr;

    bool                                                isClusterInfoChanged = true;
    uint32_t                                            maxClusterEdgeSegments = 0;

    // Buffers used to create templates.They are created once but need to be persistent throughout the app's run time.
    CUdeviceptr                                         d_templateBuffer     = 0;           // grid template buffer
    CUdeviceptr*                                        d_templatePtrsBuffer = nullptr;     // address of each template in the template buffer
    uint32_t*                                           d_templateSizeData   = nullptr;     // template size buffer holds the worst case size of a cluster that can be created for each template

    // below variables are defined here for avoiding re-alloc every frame
    size_t*                                             d_clasAddressOffsets = nullptr;     // CLAS address offsets related to the big CLAS buffer 
    CUdeviceptr                                         d_tempBuffer         = 0;           // temp buffer used in cluster and flat triangle build
    size_t                                              tempBufferSize       = 0; 
    OptixClusterAccelBuildInputTemplatesArgs*           d_templatesArgsData  = nullptr;     // used in buildCLASesFromTemplates()
    OptixClusterAccelBuildInputClustersArgs*            d_clustersArgs       = nullptr;

    // IAS
    OptixInstance*                                      d_instances         = nullptr;
    OptixBuildInput                                     iasInstanceInput    = {};
    OptixTraversableHandle                              iasHandle           = 0;
    CUdeviceptr                                         d_iasTempBuffer     = 0;
    size_t                                              iasTempBufferSize   = 0;
    CUdeviceptr                                         d_iasOutputBuffer   = 0;
    size_t                                              iasOutputBufferSize = 0;

    OptixModule                                         ptxModule              = 0;
    OptixPipelineCompileOptions                         pipelineCompileOptions = {};
    OptixPipeline                                       pipeline               = 0;

    OptixProgramGroup                                   raygenProgGroup = 0;
    OptixProgramGroup                                   missGroup       = 0;
    OptixProgramGroup                                   hitGroup        = 0;
    OptixShaderBindingTable                             sbt             = {};

    Params                                              params   = {};
    Params*                                             d_params = 0;

    float                                               time = 0.f;
};


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
    g_mousePos     = make_float2( float( xpos ), float( ypos ) );
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
    else if( key == GLFW_KEY_1 )
    {
        g_edgeSegments = make_uint2( std::max( 1u, --g_edgeSegments.x ) );
        g_isClusterInfoChanged = true;
        printf( "cluster edge segment: %dx%d\n", g_edgeSegments.x, g_edgeSegments.y );
    }
    else if( key == GLFW_KEY_2 )
    {
        g_edgeSegments = make_uint2( std::min( g_maxClusterEdgeSegments, ++g_edgeSegments.x ) );
        g_isClusterInfoChanged = true;
        printf( "cluster edge segment: %dx%d\n", g_edgeSegments.x, g_edgeSegments.y );
    }
    else if( key == GLFW_KEY_3 )
    {
        g_gridDims = make_uint2( std::max( 1u, g_gridDims.x / 2 ) );
        g_isClusterInfoChanged = true;
        printf( "grid dimension: %dx%d\n", g_gridDims.x, g_gridDims.y );
    }
    else if( key == GLFW_KEY_4 )
    {
        g_gridDims = make_uint2( std::min( 1024u, g_gridDims.x * 2 ) );
        g_isClusterInfoChanged = true;
        printf( "grid dimension:: %dx%d\n", g_gridDims.x, g_gridDims.y );
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
    std::cerr << "         --es=<ex>x<ey>                   Number of cluster edge segments for horizontal and vertical edges (default 8 8) min: 1 1, max: 11 11\n";
    std::cerr << "         --gd=<sx>x<sy>                   Grid dimensions (default 512 512) min: 1 1, max: 1024 1024\n";
    std::cerr << "         --time | -t                      Animation time for image output (default 1)\n";
    std::cerr << "         --frames | -n                    Number of animation frames for image output (default 16)\n";
    std::cerr << "         --no-gl-interop                  Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>           Set image dimensions; defaults to 1024x768\n";
    std::cerr << "         --help | -h                      Print this usage message\n";
    std::cerr << "\n";
    exit( 0 );
}


void initLaunchParams( StructuredClusterState& state )
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

    float2 mousePosNorm = g_mousePos / make_float2( float( params.width ), float( params.height ) ) * 2.f - 1.f;
    mousePosNorm.y      = -mousePosNorm.y;  // Flip vertically since screen y is opposite world y
    g_mousePosInWorld   = params.eye + params.W + mousePosNorm.x * params.U + mousePosNorm.y * params.V;

    handleCameraUpdate( params );
    handleResize( outputBuffer, params );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& outputBuffer, StructuredClusterState& state )
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

    OPTIX_CHECK( optixLaunch( 
        state.pipeline, state.stream, 
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


void initCameraState( StructuredClusterState& state )
{
    sutil::Aabb scene_aabb;
    scene_aabb.invalidate();
    for( const auto& instance : state.scene.instances() )
        scene_aabb.include( instance->world_aabb );

    g_camera.setLookat( scene_aabb.center() );
    g_camera.setEye( scene_aabb.center() + make_float3( 1.1f ) * scene_aabb.maxExtent() );
    g_camera.setFovY( 35.0f );
    g_cameraChanged = true;

    g_trackball.setCamera( &g_camera );
    g_trackball.setMoveSpeed( 10.0f );
    g_trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    g_trackball.setGimbalLock( true );
}


void createContext( StructuredClusterState& state )
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( 0 ) );

    CUcontext cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 2;
    options.validationMode            = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &state.context ) );
}


//------------------------------------------------------------------------------
//
// Helper functions for creating templates, CLAS etc 
//
//------------------------------------------------------------------------------
void inline resizeDeviceBuffer( CUdeviceptr& inOuputBuffer, size_t& oldSizeInByte, size_t newSizeInByte, CUstream stream = 0 )
{
    if( oldSizeInByte < newSizeInByte )
    {
        CUdeviceptr tmpBuffer = 0;
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &tmpBuffer ), newSizeInByte, stream ) );
        CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( tmpBuffer ), reinterpret_cast<void*>( inOuputBuffer ),
                                     oldSizeInByte, cudaMemcpyDeviceToDevice, stream ) );
        std::swap( inOuputBuffer, tmpBuffer );
        std::swap( oldSizeInByte, newSizeInByte );
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( tmpBuffer ), stream ) );
    }
}

// create cluster templates from grids 1x1, ... 1x11, 2x1 ... 11x11,
// it is only called once, templates are used later for creating CLAS
void inline initStructuredClusterTemplates( StructuredClusterState& state )
{
    const uint32_t maxNumTemplates     = state.maxClusterEdgeSegments * state.maxClusterEdgeSegments;
    const uint32_t clusterMaxTriangles = state.maxClusterEdgeSegments * state.maxClusterEdgeSegments * 2;
    const uint32_t clusterMaxVertices  = ( state.maxClusterEdgeSegments + 1 ) * ( state.maxClusterEdgeSegments + 1 );

    /*------------ Create Templates PrebuildInfo-------------*/
    OptixClusterAccelBuildInput clusterBuildInput  = {};
    clusterBuildInput.type                        = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_GRIDS;
    clusterBuildInput.grids.flags                 = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
    clusterBuildInput.grids.maxArgCount           = maxNumTemplates;
    clusterBuildInput.grids.vertexFormat          = OPTIX_VERTEX_FORMAT_FLOAT3;
    clusterBuildInput.grids.maxSbtIndexValue      = 0;
    clusterBuildInput.grids.maxWidth              = state.maxClusterEdgeSegments;
    clusterBuildInput.grids.maxHeight             = state.maxClusterEdgeSegments;

    OptixAccelBufferSizes templateBufferSizes{};
    OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( state.context, OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
                                                      &clusterBuildInput, &templateBufferSizes ) );

    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_templateBuffer ),
                                 templateBufferSizes.outputSizeInBytes, state.stream ) );
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_templatePtrsBuffer ),
                                 maxNumTemplates * sizeof( CUdeviceptr ), state.stream ) );
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_templateSizeData ),
                                 maxNumTemplates * sizeof( uint32_t ), state.stream ) );

    /*------------ Create Templates -------------*/
    resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, templateBufferSizes.tempSizeInBytes, state.stream );
    CUDA_CHECK( cudaMemsetAsync( reinterpret_cast<void*>( state.d_tempBuffer ), 0, state.tempBufferSize, state.stream ) );

    OptixClusterAccelBuildInputGridsArgs* d_inputGridsArgs = nullptr;
    {
        std::vector<OptixClusterAccelBuildInputGridsArgs> inputGridsArgsData( maxNumTemplates, { 0 } );
        for( uint32_t tId = 0; tId < maxNumTemplates; tId++ )
        {
            OptixClusterAccelBuildInputGridsArgs& inputGridsArg = inputGridsArgsData[tId];
            inputGridsArg.basePrimitiveInfo.primitiveFlags      = (unsigned int)OPTIX_CLUSTER_ACCEL_PRIMITIVE_FLAG_DISABLE_ANYHIT;
            inputGridsArg.dimensions[0]                         = (uint8_t)( tId / state.maxClusterEdgeSegments + 1 );
            inputGridsArg.dimensions[1]                         = (uint8_t)( tId % state.maxClusterEdgeSegments + 1 );
        }
        size_t inputGridsArgsSize = maxNumTemplates * sizeof( OptixClusterAccelBuildInputGridsArgs );
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_inputGridsArgs ), inputGridsArgsSize, state.stream ) );
        CUDA_CHECK( cudaMemcpyAsync( d_inputGridsArgs, inputGridsArgsData.data(), inputGridsArgsSize,
                                     cudaMemcpyHostToDevice, state.stream ) );
    }

    OptixClusterAccelBuildModeDesc clusterAccelBuildModeDec          = {};
    clusterAccelBuildModeDec.mode                                    = OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
    clusterAccelBuildModeDec.implicitDest.outputBuffer               = state.d_templateBuffer;
    clusterAccelBuildModeDec.implicitDest.outputBufferSizeInBytes    = templateBufferSizes.outputSizeInBytes;
    clusterAccelBuildModeDec.implicitDest.tempBuffer                 = state.d_tempBuffer;
    clusterAccelBuildModeDec.implicitDest.tempBufferSizeInBytes      = state.tempBufferSize;
    clusterAccelBuildModeDec.implicitDest.outputHandlesBuffer        = reinterpret_cast<CUdeviceptr>( state.d_templatePtrsBuffer );
    clusterAccelBuildModeDec.implicitDest.outputHandlesStrideInBytes = sizeof( CUdeviceptr );

    OPTIX_CHECK( optixClusterAccelBuild( state.context, state.stream, &clusterAccelBuildModeDec, &clusterBuildInput,
                                         reinterpret_cast<CUdeviceptr>( d_inputGridsArgs ), 0 /*argsCount*/,
                                         static_cast<uint32_t>( sizeof( OptixClusterAccelBuildInputGridsArgs ) ) ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( d_inputGridsArgs ), state.stream ) );

    /*------------ Compute Templates Sizes -------------*/
    // Instantiate templates args without providing the vertex positions returns the worst case size for each grid/cluster size
    OptixClusterAccelBuildInputTemplatesArgs* d_templatesArgs = nullptr;
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_templatesArgs ),
                                 maxNumTemplates * sizeof( OptixClusterAccelBuildInputTemplatesArgs ), state.stream ) );
    makeTemplatesArgsDataForGetSizes( state.stream, state.d_templatePtrsBuffer, maxNumTemplates, d_templatesArgs );

    OptixClusterAccelBuildInput getTemplateSizesBuildInputs            = {};
    getTemplateSizesBuildInputs.type                                   = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES;
    getTemplateSizesBuildInputs.triangles.flags                        = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
    getTemplateSizesBuildInputs.triangles.maxArgCount                  = maxNumTemplates;
    getTemplateSizesBuildInputs.triangles.vertexFormat                 = OPTIX_VERTEX_FORMAT_FLOAT3;
    getTemplateSizesBuildInputs.triangles.maxUniqueSbtIndexCountPerArg = 1;
    getTemplateSizesBuildInputs.triangles.maxTriangleCountPerArg       = clusterMaxTriangles;
    getTemplateSizesBuildInputs.triangles.maxVertexCountPerArg         = clusterMaxVertices;
    getTemplateSizesBuildInputs.triangles.maxTotalTriangleCount        = clusterMaxTriangles * maxNumTemplates;
    getTemplateSizesBuildInputs.triangles.maxTotalVertexCount          = clusterMaxVertices * maxNumTemplates;

    OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( state.context, OPTIX_CLUSTER_ACCEL_BUILD_MODE_GET_SIZES,
                                                      &getTemplateSizesBuildInputs, &templateBufferSizes ) );

    resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, templateBufferSizes.tempSizeInBytes, state.stream );
    CUDA_CHECK( cudaMemsetAsync( reinterpret_cast<void*>( state.d_tempBuffer ), 0, state.tempBufferSize, state.stream ) );

    OptixClusterAccelBuildModeDesc getTemplateSizesBuildDesc   = {};
    getTemplateSizesBuildDesc.mode                             = OPTIX_CLUSTER_ACCEL_BUILD_MODE_GET_SIZES;
    getTemplateSizesBuildDesc.getSize.tempBuffer               = state.d_tempBuffer;
    getTemplateSizesBuildDesc.getSize.tempBufferSizeInBytes    = state.tempBufferSize;
    getTemplateSizesBuildDesc.getSize.outputSizesBuffer        = reinterpret_cast<CUdeviceptr>( state.d_templateSizeData );
    getTemplateSizesBuildDesc.getSize.outputSizesStrideInBytes = sizeof( uint32_t );

    OPTIX_CHECK( optixClusterAccelBuild( state.context, state.stream, &getTemplateSizesBuildDesc, &getTemplateSizesBuildInputs,
                                         reinterpret_cast<CUdeviceptr>( d_templatesArgs ), 0 /*argsCount*/, 0 ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( d_templatesArgs ), state.stream ) );
}


// build CLAS from the pre-generated templates, instantiate the templates with vertex data
void inline buildCLASesFromTemplates( StructuredClusterState& state )
{
    ClusterAccel* accel = reinterpret_cast<ClusterAccel*>( state.pAccel.get() );
    if( state.isClusterInfoChanged )
    {
        // cluster info change will result in the clusterCount change, need to re-alloc the buffers.

        resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, sizeof( size_t ), state.stream );
        CUDA_CHECK( cudaMemsetAsync( (void*)state.d_tempBuffer, 0, state.tempBufferSize, state.stream ) );
        size_t* d_outputSizeInBytes = reinterpret_cast<size_t*>( state.d_tempBuffer );

        if( state.d_clasAddressOffsets )
        {
            CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clasAddressOffsets ), state.stream ) );
        }
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clasAddressOffsets ),
                                     state.clusterCount * sizeof( size_t ), state.stream ) );

        calculateClasOutputBufferSizeAndOffsets(
            state.stream,
            state.clusterCount, 
            state.maxClusterEdgeSegments,
            reinterpret_cast<uchar2*>( (unsigned char*)state.d_clusters + offsetof( Cluster, size ) ), 
            state.d_templateSizeData,
            sizeof( Cluster ), 
            d_outputSizeInBytes,
            state.d_clasAddressOffsets
        );

        size_t clasSizeInBytes = 0;
        CUDA_CHECK( cudaMemcpyAsync( &clasSizeInBytes, d_outputSizeInBytes, sizeof( size_t ), cudaMemcpyDeviceToHost,
                                     state.stream ) );

        if( accel->d_clasBuffer )
        {
            CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( accel->d_clasBuffer ), state.stream ) );
        }
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &accel->d_clasBuffer ), clasSizeInBytes, state.stream ) );

        if( state.d_templatesArgsData )
        {
            CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_templatesArgsData ), state.stream ) );
        }
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_templatesArgsData ),
                                     state.clusterCount * sizeof( OptixClusterAccelBuildInputTemplatesArgs ), state.stream ) );

        if( accel->d_clasPtrsBuffer )
        {
            CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( accel->d_clasPtrsBuffer ), state.stream ) );
        }
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &accel->d_clasPtrsBuffer ),
                                     state.clusterCount * sizeof( CUdeviceptr ), state.stream ) );
        CUDA_CHECK( cudaMemsetAsync( reinterpret_cast<void*>( accel->d_clasPtrsBuffer ), 0,
                                     state.clusterCount * sizeof( CUdeviceptr ), state.stream ) );
    }

    const std::vector<std::shared_ptr<sutil::Scene::MeshGroup>>& meshes = state.scene.meshes();
    makeTemplatesArgsData( state.stream, state.d_clusters, state.clusterCount, state.maxClusterEdgeSegments, state.d_templatePtrsBuffer,
                           reinterpret_cast<float3*>( meshes[0]->positions[0].data ), state.d_templatesArgsData );

    const uint32_t clusterMaxTriangles = state.maxClusterEdgeSegments * state.maxClusterEdgeSegments * 2;
    const uint32_t clusterMaxVertices  = ( state.maxClusterEdgeSegments + 1 ) * ( state.maxClusterEdgeSegments + 1 );

    OptixClusterAccelBuildInput clasBuildInputs            = {};
    clasBuildInputs.type                                   = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES;
    clasBuildInputs.triangles.flags                        = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
    clasBuildInputs.triangles.maxArgCount                  = state.clusterCount;
    clasBuildInputs.triangles.vertexFormat                 = OPTIX_VERTEX_FORMAT_FLOAT3;
    clasBuildInputs.triangles.maxUniqueSbtIndexCountPerArg = 1;
    clasBuildInputs.triangles.maxTriangleCountPerArg       = clusterMaxTriangles;
    clasBuildInputs.triangles.maxVertexCountPerArg         = clusterMaxVertices;
    clasBuildInputs.triangles.maxTotalTriangleCount        = clusterMaxTriangles * state.clusterCount;
    clasBuildInputs.triangles.maxTotalVertexCount          = clusterMaxVertices * state.clusterCount;

    OptixAccelBufferSizes tempBufferSizes = {};
    OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( state.context, OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
                                                      &clasBuildInputs, &tempBufferSizes ) );

    resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, tempBufferSizes.tempSizeInBytes, state.stream );
    CUDA_CHECK( cudaMemsetAsync( (void*)state.d_tempBuffer, 0, state.tempBufferSize, state.stream ) );

    assignExplicitAddresses( state.stream, state.clusterCount, state.d_clasAddressOffsets, accel->d_clasBuffer, accel->d_clasPtrsBuffer );

    // Here need to be EXPLICIT mode becuase we use the size estimates from a GET_SIZES call earlier. 
    // GET_SIZES is only valid to be used with explicit builds.
    OptixClusterAccelBuildModeDesc clasBuildDesc          = {};
    clasBuildDesc.mode                                    = OPTIX_CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS;
    clasBuildDesc.explicitDest.destAddressesBuffer        = reinterpret_cast<CUdeviceptr>( accel->d_clasPtrsBuffer );
    clasBuildDesc.explicitDest.destAddressesStrideInBytes = sizeof(CUdeviceptr);
    clasBuildDesc.explicitDest.tempBuffer                 = state.d_tempBuffer;
    clasBuildDesc.explicitDest.tempBufferSizeInBytes      = state.tempBufferSize;
    clasBuildDesc.explicitDest.outputHandlesBuffer        = reinterpret_cast<CUdeviceptr>( accel->d_clasPtrsBuffer );
    clasBuildDesc.explicitDest.outputHandlesStrideInBytes = sizeof( CUdeviceptr );

    OPTIX_CHECK( optixClusterAccelBuild( state.context, state.stream, &clasBuildDesc, &clasBuildInputs,
                                         reinterpret_cast<CUdeviceptr>( state.d_templatesArgsData ), 0 /*argsCount*/, 0 ) );
}


void inline buildGASesFromCLASes( StructuredClusterState& state )
{
    const std::vector<std::shared_ptr<sutil::Scene::Instance>>& instances = state.scene.instances();
    const uint32_t                                              instanceCount = (uint32_t)instances.size();
    assert( instanceCount == 1 ); // will not change in this sample 

    std::vector<size_t> clusterOffsets( instanceCount + 1u, 0 );
    if( !state.d_clusterOffsets )
    {
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clusterOffsets ),
                                     clusterOffsets.size() * sizeof( size_t ), state.stream ) );
    }

    clusterOffsets[1] = state.clusterCount;
    CUDA_CHECK( cudaMemcpyAsync( state.d_clusterOffsets, clusterOffsets.data(),
                                 clusterOffsets.size() * sizeof( size_t ), cudaMemcpyHostToDevice, state.stream ) );

    const uint32_t maxClusterCount        = state.clusterCount;
    const uint32_t maxClusterCountPerBlas = state.clusterCount;

    OptixClusterAccelBuildInput gasBuildInputs    = {};
    gasBuildInputs.type                           = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS;
    gasBuildInputs.clusters.flags                 = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
    gasBuildInputs.clusters.maxArgCount           = instanceCount;
    gasBuildInputs.clusters.maxTotalClusterCount  = maxClusterCount;
    gasBuildInputs.clusters.maxClusterCountPerArg = maxClusterCountPerBlas;

    OptixAccelBufferSizes gasBufferSizes{};
    OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( state.context, OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
                                                      &gasBuildInputs, &gasBufferSizes ) );

    resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, gasBufferSizes.tempSizeInBytes, state.stream );

    ClusterAccel* accel = reinterpret_cast<ClusterAccel*>( state.pAccel.get() );
    resizeDeviceBuffer( accel->d_gasBuffer, accel->gasBufferSize, gasBufferSizes.outputSizeInBytes, state.stream );


    // only need to allocate once since instanceCount will never update in this sample.
    if( !accel->d_gasPtrsBuffer)
    {
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &accel->d_gasPtrsBuffer ),
                                     instanceCount * sizeof( CUdeviceptr ), state.stream ) );
    }

    if( !accel->d_gasSizesBuffer )
    {
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &accel->d_gasSizesBuffer ),
                                     instanceCount * sizeof( uint32_t ), state.stream ) );
    }

    if( !state.d_clustersArgs )
    {
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clustersArgs ),
                                     instanceCount * sizeof( OptixClusterAccelBuildInputClustersArgs ), state.stream ) );
    }
    makeClustersArgsData( state.stream, state.d_clusterOffsets, instanceCount, accel->d_clasPtrsBuffer, state.d_clustersArgs );

    OptixClusterAccelBuildModeDesc gasBuildDesc          = {};
    gasBuildDesc.mode                                    = OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
    gasBuildDesc.implicitDest.outputBuffer               = accel->d_gasBuffer;
    gasBuildDesc.implicitDest.outputBufferSizeInBytes    = accel->gasBufferSize;
    gasBuildDesc.implicitDest.tempBuffer                 = state.d_tempBuffer;
    gasBuildDesc.implicitDest.tempBufferSizeInBytes      = state.tempBufferSize;
    gasBuildDesc.implicitDest.outputHandlesBuffer        = reinterpret_cast<CUdeviceptr>( accel->d_gasPtrsBuffer );
    gasBuildDesc.implicitDest.outputHandlesStrideInBytes = sizeof( CUdeviceptr );
    gasBuildDesc.implicitDest.outputSizesBuffer          = reinterpret_cast<CUdeviceptr>( accel->d_gasSizesBuffer );
    gasBuildDesc.implicitDest.outputSizesStrideInBytes   = sizeof( uint32_t );

    OPTIX_CHECK( optixClusterAccelBuild( state.context, state.stream, &gasBuildDesc, &gasBuildInputs,
                                         reinterpret_cast<CUdeviceptr>( state.d_clustersArgs ), 0 /*argsCount*/, 0 ) );
}


void inline buildClusterAccel( StructuredClusterState& state )
{
    if( state.isClusterInfoChanged )
    {
        if( state.d_clusters )
            CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clusters ), state.stream ) );

        uint32_t newClusterCount = state.tessConfig.gridDims.x * state.tessConfig.gridDims.y;
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_clusters ), newClusterCount * sizeof( Cluster ),
                                     state.stream ) );
        makeClusters( state.stream, state.tessConfig, state.d_clusters );  // tessellate a square surface into multiple clusters and fill the metadata (size, vtxOffset)
        state.clusterCount = newClusterCount;
    }

    const uint32_t trianglesPerCluster = 2 * state.tessConfig.clusterSize.x * state.tessConfig.clusterSize.y;
    state.totalTriangleCount           = state.clusterCount * trianglesPerCluster;

    const std::vector<std::shared_ptr<sutil::Scene::MeshGroup>>& meshes = state.scene.meshes();
    assert( meshes.size() == 1 && meshes[0]->positions.size() == 1 );

    const uint32_t numVerticesPerCluster = ( state.tessConfig.clusterSize.x + 1 ) * ( state.tessConfig.clusterSize.y + 1 );
    const uint32_t numVertices           = state.clusterCount * numVerticesPerCluster;

    std::shared_ptr<sutil::Scene::MeshGroup> mesh = meshes[0];
    if( mesh->positions[0].count != numVertices )
    {
        // vertex buffer re-allocated only if the tessConfig is changed.
        mesh->positions[0].count = numVertices;
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( mesh->positions[0].data ), state.stream ) );
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &mesh->positions[0].data ),
                                     mesh->positions[0].count * sizeof( float3 ), state.stream ) );
    }
    // generate the cluster vertices buffer
    generateAnimatedVertices( state.stream, state.tessConfig, mesh->positions[0].data );

    buildCLASesFromTemplates( state );

    buildGASesFromCLASes( state );
}


void inline buildFlatTriangleAccel( StructuredClusterState& state )
{
    const std::vector<std::shared_ptr<sutil::Scene::MeshGroup>>& meshes = state.scene.meshes();
    std::shared_ptr<sutil::Scene::MeshGroup> mesh = meshes[0];
    assert( mesh->name == "synthetic" && meshes.size() == 1 );

    const size_t num_subMeshes = mesh->indices.size();
    assert( num_subMeshes == 1 );

    state.totalTriangleCount = mesh->indices[0].count / 3;

    FlatTriAccel* accel = dynamic_cast<FlatTriAccel*>( state.pAccel.get() );

    const uint32_t quadCount = state.tessConfig.gridDims.x * state.tessConfig.gridDims.y;
    const uint32_t numVerticesPerCluster = ( state.tessConfig.clusterSize.x + 1 ) * ( state.tessConfig.clusterSize.y + 1 );
    const uint32_t numVertices = quadCount * numVerticesPerCluster;
    if( mesh->positions[0].count != numVertices )
    {
        // vertex buffer re-allocated only if the tessConfig is changed. 
        mesh->positions[0].count = numVertices;
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( mesh->positions[0].data ), state.stream ) );
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &mesh->positions[0].data ),
                                     mesh->positions[0].count * sizeof( float3 ), state.stream ) );
    }
    // animate the vertices for the generated sythetic data. the synthetic data only have 1 position buffer, so hard code to 0
    generateAnimatedVertices( state.stream, state.tessConfig, mesh->positions[0].data );

    const uint32_t trianglesPerCluster = 2 * state.tessConfig.clusterSize.x * state.tessConfig.clusterSize.y;
    const uint32_t numIndices          = quadCount * trianglesPerCluster * 3;
    // indices re-generation only if the tessConfig is changed. 
    if( mesh->indices[0].count != numIndices )
    {
        mesh->indices[0].count = numIndices;
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( mesh->indices[0].data ), state.stream ) );
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &mesh->indices[0].data ),
                                     mesh->indices[0].count * sizeof( uint32_t ), state.stream ) );
        generateIndices( state.stream, state.tessConfig, mesh->indices[0].data );
    }

    // Build an AS over the triangles.
    std::vector<OptixBuildInput> buildInputs( num_subMeshes );
    const unsigned int           triangleFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    OptixBuildInput&             triangleInput = buildInputs[0];
    memset( &triangleInput, 0, sizeof( OptixBuildInput ) );
    triangleInput.type                        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangleInput.triangleArray.vertexFormat  = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.numVertices   = mesh->positions[0].count;
    triangleInput.triangleArray.vertexBuffers = &( mesh->positions[0].data );
    triangleInput.triangleArray.indexFormat   = mesh->indices[0].elmt_byte_size == 0 ? OPTIX_INDICES_FORMAT_NONE :
                                                mesh->indices[0].elmt_byte_size == 1 ? OPTIX_INDICES_FORMAT_UNSIGNED_BYTE3 :
                                                mesh->indices[0].elmt_byte_size == 2 ? OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 :
                                                                                       OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes =
        mesh->indices[0].byte_stride ? mesh->indices[0].byte_stride * 3 : mesh->indices[0].elmt_byte_size * 3;
    triangleInput.triangleArray.numIndexTriplets            = mesh->indices[0].count / 3;
    triangleInput.triangleArray.indexBuffer                 = mesh->indices[0].data;
    triangleInput.triangleArray.numSbtRecords               = 1;
    triangleInput.triangleArray.flags                       = &triangleFlags;
    triangleInput.triangleArray.sbtIndexOffsetBuffer        = 0;
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasBufferSizes = {};
    OPTIX_CHECK( optixAccelComputeMemoryUsage( state.context, &accelOptions, buildInputs.data(),
                                               (uint32_t)buildInputs.size(), &gasBufferSizes ) );

    resizeDeviceBuffer( state.d_tempBuffer, state.tempBufferSize, gasBufferSizes.tempSizeInBytes, state.stream );
    resizeDeviceBuffer( accel->d_gasBuffer, accel->gasBufferSize, gasBufferSizes.outputSizeInBytes, state.stream );

    OPTIX_CHECK( optixAccelBuild( 
        state.context, 
        state.stream, 
        &accelOptions, 
        buildInputs.data(),
        (uint32_t)buildInputs.size(),
        state.d_tempBuffer,
        state.tempBufferSize,
        accel->d_gasBuffer,
        accel->gasBufferSize,
        &accel->gasHandle,
        NULL,  // emitted property list
        0      // num emitted properties
    ) );
}

void buildGAS( StructuredClusterState& state )
{
    if( state.params.accelBuildMode == AccelBuildMode::CLUSTER )
    {
        if( !state.pAccel )
            state.pAccel.reset( new ClusterAccel() );
        buildClusterAccel( state );
    }
    else  // Flat triangle code path
    {
        if( !state.pAccel )
            state.pAccel.reset( new FlatTriAccel() );
        buildFlatTriangleAccel( state );
    }
}


void buildIAS( StructuredClusterState& state )
{
    // Build the IAS
    const std::vector<std::shared_ptr<sutil::Scene::Instance>>& meshInstances = state.scene.instances();
    assert( meshInstances.size() == 1 );

    std::vector<OptixInstance> instances( meshInstances.size() );
    for( size_t i = 0; i < instances.size(); ++i )
    {
        memcpy( instances[i].transform, meshInstances[i]->transform.getData(), sizeof( float ) * 12 );
        instances[i].sbtOffset      = static_cast<unsigned int>( i );
        instances[i].visibilityMask = 255;

        if( state.params.accelBuildMode == AccelBuildMode::FLAT_TRIANGLE )
        {
            instances[i].traversableHandle =
                reinterpret_cast<FlatTriAccel*>( state.pAccel.get() )->gasHandle;
        }
    }

    if( !state.d_instances )
    {
        CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &state.d_instances ),
                                     instances.size() * sizeof( OptixInstance ), state.stream ) );
    }
    CUDA_CHECK( cudaMemcpyAsync( state.d_instances, instances.data(), instances.size() * sizeof( OptixInstance ),
                                 cudaMemcpyHostToDevice, state.stream ) );

    if( state.params.accelBuildMode == AccelBuildMode::CLUSTER )
    {
        ClusterAccel* accel = reinterpret_cast<ClusterAccel*>( state.pAccel.get() );
        copyGasHandlesToInstanceArray( state.stream, state.d_instances, accel->d_gasPtrsBuffer,
                                       static_cast<uint32_t>( instances.size() ) );
    }

    state.iasInstanceInput.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    state.iasInstanceInput.instanceArray.instances    = reinterpret_cast<CUdeviceptr>( state.d_instances );
    state.iasInstanceInput.instanceArray.numInstances = static_cast<int>( instances.size() );

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
    ) );

    state.params.handle = state.iasHandle;
}

void createModule( StructuredClusterState& state )
{
    OptixModuleCompileOptions moduleCompileOptions = {};
#if OPTIX_DEBUG_DEVICE_CODE
    moduleCompileOptions.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    state.pipelineCompileOptions.usesMotionBlur        = false;
    state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    state.pipelineCompileOptions.numPayloadValues      = 3;
    state.pipelineCompileOptions.numAttributeValues    = 2;
    state.pipelineCompileOptions.exceptionFlags        = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    state.pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    state.pipelineCompileOptions.allowClusteredGeometry           = true;

    size_t      inputSize = 0;
    const char* input     = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixClusterStructuredMesh.cu", inputSize );

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


void createProgramGroups( StructuredClusterState& state )
{
    OptixProgramGroupOptions  programGroupOptions = {};

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


void createPipeline( StructuredClusterState& state )
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
        LOG, &LOG_SIZE,
        &state.pipeline
    ) );

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
    ) );

    // This is 2 since the largest depth is IAS->GAS
    const uint32_t maxTraversableGraphDepth = 2;

    OPTIX_CHECK( optixPipelineSetStackSize(
        state.pipeline,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize,
        maxTraversableGraphDepth
    ) );
}


void createSBT( StructuredClusterState& state )
{
    CUdeviceptr  d_raygenRecord;
    const size_t raygenRecordSize = sizeof( RayGenRecord );
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_raygenRecord ), raygenRecordSize, state.stream ) );

    RayGenRecord rgSBT = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( state.raygenProgGroup, &rgSBT ) );

    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_raygenRecord ), &rgSBT, raygenRecordSize,
                                 cudaMemcpyHostToDevice, state.stream ) );

    CUdeviceptr  d_missRecords;
    const size_t missRecordSize = sizeof( MissRecord );
    CUDA_CHECK( cudaMallocAsync( reinterpret_cast<void**>( &d_missRecords ), missRecordSize, state.stream ) );

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


void cleanupState( StructuredClusterState& state )
{
    state.scene.cleanup();
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clusters ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clusterOffsets ), state.stream ) );

    if( g_accelBuildMode == AccelBuildMode::CLUSTER )
    {
        ClusterAccel* accel = dynamic_cast<ClusterAccel*>( state.pAccel.get() );
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( accel->d_gasBuffer ), state.stream ) );
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( accel->d_gasPtrsBuffer ), state.stream ) );
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( accel->d_gasSizesBuffer ), state.stream ) );
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( accel->d_clasPtrsBuffer ), state.stream ) );
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( accel->d_clasBuffer ), state.stream ) );
    }
    else
    {
        // Flat triangle build
        FlatTriAccel* accel = dynamic_cast<FlatTriAccel*>( state.pAccel.get() );
        CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( accel->d_gasBuffer ), state.stream ) );
    }

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_templateBuffer ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_templatePtrsBuffer ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_templateSizeData ), state.stream ) );

    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_clasAddressOffsets ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_tempBuffer ), state.stream ) );
    CUDA_CHECK( cudaFreeAsync( reinterpret_cast<void*>( state.d_templatesArgsData ), state.stream ) );
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

void displayStats( StructuredClusterState& state, float& accelBuildTime, float& renderTime )
{
    constexpr std::chrono::duration<double> displayUpdateMinIntervalTime( 0.5 );
    static int32_t                          totalSubframeCount = 0;
    static int32_t                          lastUpdateFrames   = 0;
    static auto                             lastUpdateTime     = std::chrono::steady_clock::now();
    static char                             displayText[256];

    const auto curTime = std::chrono::steady_clock::now();

    sutil::beginFrameImGui();
    lastUpdateFrames++;

    if( curTime - lastUpdateTime > displayUpdateMinIntervalTime || totalSubframeCount == 0 )
    {
        sprintf( displayText,
                 "%5.1f fps\n\n"
                 "tri count   : %8.2f M\n"
                 "gas build   : %8.2f ms\n"
                 "render      : %8.2f ms\n"
                 "cluster size:  \t%d x %d\n"
                 "grid dims   :  \t%d x %d\n\n"
                 "key:\n"
                 "\t1/2: decrease/increase cluster size\n"
                 "\t3/4: decrease/increase grid dimension\n",
                 lastUpdateFrames / std::chrono::duration<double>( curTime - lastUpdateTime ).count(),
                 static_cast<float>( state.totalTriangleCount / 1000 / 1000 ), 
                 accelBuildTime / lastUpdateFrames,
                 renderTime / lastUpdateFrames, 
                 state.tessConfig.clusterSize.x,
                 state.tessConfig.clusterSize.y, 
                 state.tessConfig.gridDims.x, 
                 state.tessConfig.gridDims.y );

        lastUpdateTime   = curTime;
        lastUpdateFrames = 0;
        accelBuildTime = renderTime = 0.f;
    }

    sutil::displayText( displayText, 10.0f, 10.0f );

    int         selectedOption  = static_cast<int>( g_accelBuildMode );
    const char* options[]       = { "Cluster build", "Flat triangle build" };
    sutil::buildRadioButtons( options, IM_ARRAYSIZE( options ), state.params.width - 200.f, 10.f, selectedOption );
    if( selectedOption != static_cast<int>( g_accelBuildMode ) )
    {
        g_accelBuildMode          = static_cast<AccelBuildMode>( selectedOption );
        g_isAccelBuildModeChanged = true;
    }
    sutil::endFrameImGui();

    ++totalSubframeCount;
}


void makeSyntheticSceneData( StructuredClusterState& state )
{
    // synthetic grid data for structured cluster data, it has only 1 mesh
    std::shared_ptr<sutil::Scene::MeshGroup> mesh = std::make_shared<sutil::Scene::MeshGroup>();
    state.scene.addMesh( mesh );
    mesh->name = "synthetic";
    mesh->object_aabb.invalidate();
    mesh->object_aabb.include( sutil::Aabb( make_float3( -1.f, 0.f, -1.f ), make_float3( 1.f, 0.f, 1.f ) ) );

    BufferView<float3> verticesBufferView;
    verticesBufferView.byte_stride    = static_cast<uint16_t>( 0 );
    verticesBufferView.elmt_byte_size = static_cast<uint16_t>( sizeof( float ) );
    mesh->positions.push_back( verticesBufferView );

    GenericBufferView indicesBufferView;
    indicesBufferView.byte_stride    = static_cast<uint16_t>( 0 );
    indicesBufferView.elmt_byte_size = static_cast<uint16_t>( sizeof( uint32_t ) );
    mesh->indices.push_back( indicesBufferView );

    // synthetic grid  has only 1 instance
    std::shared_ptr<sutil::Scene::Instance> instance = std::make_shared<sutil::Scene::Instance>();
    instance->transform  = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    instance->mesh_idx   = 0;
    instance->world_aabb = state.scene.meshes()[instance->mesh_idx]->object_aabb;
    state.scene.addInstance( instance );
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    StructuredClusterState state;
    state.params.width                             = 1024;
    state.params.height                            = 768;
    state.time                                     = 0.f;
    sutil::CUDAOutputBufferType outputBufferType   = sutil::CUDAOutputBufferType::GL_INTEROP;

    g_mousePos = make_float2( state.params.width * 0.5f, state.params.height * 0.5f );

    int   numFrames     = 16;
    float animationTime = 1.f;

    try
    {
        //
        // Set up OptiX state, check if supporting cluster acceleration struture and quer device property
        //
        createContext( state );

        int isClustersSupported = 0;
        OPTIX_CHECK( optixDeviceContextGetProperty( state.context, OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL,
                                                    &isClustersSupported, sizeof( isClustersSupported ) ) );
        if( !isClustersSupported )
        {
            std::cerr << "Cluster API is not supported on this GPU" << std::endl;
            exit( 1 );
        }

        OPTIX_CHECK( optixDeviceContextGetProperty( state.context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_STRUCTURED_GRID_RESOLUTION,
                                                    &g_maxClusterEdgeSegments, sizeof( g_maxClusterEdgeSegments ) ) );
        state.maxClusterEdgeSegments = g_maxClusterEdgeSegments;
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    //
    // Parse command line options
    //
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
        else if( arg == "--time" || arg == "-t" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            animationTime = (float)atof( argv[++i] );
        }
        else if( arg == "--frames" || arg == "-n" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );

            numFrames = atoi( argv[++i] );
        }
        else if( arg.substr( 0, 5 ) == "--es=" )
        {
            const std::string edgeSegmentsArg = arg.substr( 5 );
            int               es_x, es_y;
            sutil::parseDimensions( edgeSegmentsArg.c_str(), es_x, es_y );

            if( es_x < 1 || es_x > (int)g_maxClusterEdgeSegments || es_y < 1 || es_y > (int)g_maxClusterEdgeSegments )
                printUsageAndExit( argv[0] );
            g_edgeSegments = { (uint32_t)es_x, (uint32_t)es_y };
        }
        else if( arg.substr( 0, 5 ) == "--gd=" )
        {
            const std::string gridDim_arg = arg.substr( 5 );
            int               width, height;
            sutil::parseDimensions( gridDim_arg.c_str(), width, height );
            g_gridDims = { (uint32_t)width, (uint32_t)height };
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    try
    {
        makeSyntheticSceneData( state );

        initCameraState( state );
        createModule( state );
        createProgramGroups( state );
        createPipeline( state );
        createSBT( state );
        initLaunchParams( state );

        initStructuredClusterTemplates( state );

        state.pAccel.reset( new ClusterAccel() );

        // create CUDA events for timing
        cudaEvent_t accelBuildTimeStart, accelBuildTimeStop, renderTimeStart, renderTimeStop;
        CUDA_CHECK( cudaEventCreate( &accelBuildTimeStart ) );
        CUDA_CHECK( cudaEventCreate( &accelBuildTimeStop ) );
        CUDA_CHECK( cudaEventCreate( &renderTimeStart ) );
        CUDA_CHECK( cudaEventCreate( &renderTimeStop ) );

        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixClusterStructuredMesh", state.params.width, state.params.height );
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
                    state.time                         = (float)time.count();
                    state.tessConfig.mousePosInWorld   = g_mousePosInWorld;
                    state.tessConfig.clusterSize       = g_edgeSegments;
                    state.tessConfig.gridDims          = g_gridDims;
                    state.tessConfig.animationTime     = state.time;
                    state.params.accelBuildMode        = g_accelBuildMode;
                    state.params.gridDims              = g_gridDims;

                    if( g_isAccelBuildModeChanged )
                    {
                        state.pAccel.reset();
                        g_isAccelBuildModeChanged = false;

                        // pAccel is reset when we toggle from flat triangle build mode into cluster build mode. 
                        // need to set the status here to allow re-calculating the CLAS output buffer size again once. 
                        if( g_accelBuildMode == AccelBuildMode::CLUSTER )
                        {
                            g_isClusterInfoChanged = true;
                        }
                    }

                    CUDA_CHECK( cudaEventRecord( accelBuildTimeStart, state.stream ) );
                    state.isClusterInfoChanged = g_isClusterInfoChanged;
                    buildGAS( state );
                    buildIAS( state );
                    CUDA_CHECK( cudaEventRecord( accelBuildTimeStop, state.stream ) );

                    g_isClusterInfoChanged = false;

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

                // run animation frames
                for( unsigned int i = 0; i < static_cast<unsigned int>( numFrames ); ++i )
                {
                    state.time = i * ( animationTime / ( numFrames - 1 ) );
                    state.params.accelBuildMode = g_accelBuildMode;
                    buildGAS( state );
                    buildIAS( state );
                    launchSubframe( outputBuffer, state );
                }

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
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
