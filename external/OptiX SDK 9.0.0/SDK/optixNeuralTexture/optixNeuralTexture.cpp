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

#include <glad/glad.h> // Needs to be included before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <libntc/ntc.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/Light.h>

#include <sutil/Camera.h>
#include <sutil/Trackball.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
#include <sutil/Record.h>
#include <sutil/Scene.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include <GLFW/glfw3.h>

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

#include <libntc/shaders/DecompressConstants.h>
#include "optixNeuralTexture.h"

#include "NtcTextureSet.h"


bool resize_dirty = false;
bool minimized    = false;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;

LaunchParams* d_params = nullptr;
LaunchParams  g_params = {};
int32_t       width    = 768;
int32_t       height   = 768;

OptixShaderBindingTable g_sbt = {};

OptixProgramGroup      g_raygen_prog_group    = 0;
OptixProgramGroup      g_radiance_miss_group  = 0;
OptixProgramGroup      g_occlusion_miss_group = 0;
OptixProgramGroup      g_radiance_hit_group   = 0;
OptixProgramGroup      g_occlusion_hit_group  = 0;

typedef sutil::Record<HitGroupData> HitGroupRecord;

static ntc::IContext* g_ntcContext;

uint32_t g_max_reg_count = 168;

struct Span 
{ 
    uint8_t* data;
    size_t size;
};


//---------------------------------------------------------------------------------
//
// NTC helper functions
//
//---------------------------------------------------------------------------------

void checkNtcStatus( ntc::Status ntcStatus )
{
    if( ntcStatus != ntc::Status::Ok && ntcStatus != ntc::Status::CudaUnavailable )
        printf( "NTC Error: %s, %s\n", ntc::StatusToString( ntcStatus ), ntc::GetLastErrorMessage() );
}


ntc::IContext* makeNtcContext()
{
    ntc::IContext*         context = nullptr;
    ntc::ContextParameters contextParams;
    contextParams.cudaDevice                    = 0;
    contextParams.graphicsApi                   = ntc::GraphicsAPI::None;
    contextParams.graphicsDeviceSupportsDP4a    = true;
    contextParams.graphicsDeviceSupportsFloat16 = true;
    contextParams.enableCooperativeVectorFP8    = true;
    contextParams.enableCooperativeVectorInt8   = true;

    checkNtcStatus( ntc::CreateContext( &context, contextParams ) );
    return context;
}


Span getWeightsData( const std::string& filepath, ntc::ITextureSetMetadata* metaData, ntc::InferenceWeightType weightType )
{
    const void* weightData = nullptr;
    size_t      weightSize = 0;
    checkNtcStatus( metaData->GetInferenceWeights( weightType, &weightData, &weightSize ) );
    return { (uint8_t*)weightData, weightSize };
}


std::vector<int> getLayerSizes( int networkVersion, int numHiddenLayers )
{
    const int INPUT_CHANNELS[5] = {
        0,
        NTC_MLP_INPUT_CHANNELS_SMALL,
        NTC_MLP_INPUT_CHANNELS_MEDIUM,
        NTC_MLP_INPUT_CHANNELS_LARGE,
        NTC_MLP_INPUT_CHANNELS_XLARGE
    };

    std::vector<int> layers( numHiddenLayers + 2, NTC_MLP_HIDDEN_CHANNELS );
    layers[0]     = INPUT_CHANNELS[networkVersion];
    layers.back() = NTC_MLP_OUTPUT_CHANNELS;
    return layers;
}


uint8_t* convertWeights( const ntc::ITextureSetMetadata* metaData,
                         const ntc::InferenceData&       inferenceData,
                         bool                            enableFP8,
                         const Span                      h_weights,
                         OptixDeviceContext              optix_context )
{
    const int        networkVersion  = metaData->GetNetworkVersion();
    const int        numHiddenLayers = 3;
    std::vector<int> channels        = getLayerSizes( networkVersion, numHiddenLayers );
    int              n_layers        = (int)channels.size() - 1;

    std::vector<OptixCoopVecMatrixDescription> srcLayerDesc( n_layers, OptixCoopVecMatrixDescription{} );
    std::vector<OptixCoopVecMatrixDescription> dstLayerDesc( n_layers, OptixCoopVecMatrixDescription{} );

    OptixCoopVecMatrixLayout srcMatrixLayout = OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR;
    OptixCoopVecMatrixLayout dstMatrixLayout = OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL;

    OptixCoopVecElemType layerType[4];
    if( enableFP8 )
    {
        layerType[0] = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3;
        layerType[1] = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3;
        layerType[2] = OPTIX_COOP_VEC_ELEM_TYPE_FLOAT8_E4M3;
        layerType[3] = OPTIX_COOP_VEC_ELEM_TYPE_INT8;
    }
    else
    {
        layerType[0] = OPTIX_COOP_VEC_ELEM_TYPE_INT8;
        layerType[1] = OPTIX_COOP_VEC_ELEM_TYPE_INT8;
        layerType[2] = OPTIX_COOP_VEC_ELEM_TYPE_INT8;
        layerType[3] = OPTIX_COOP_VEC_ELEM_TYPE_INT8;
    }

    const int optRowColumnStrideInBytes = 0;

    for( int i = 0; i < n_layers; i++ )
    {
        const int CIN  = channels[i];      // K
        const int COUT = channels[i + 1];  // N

        const unsigned int N = COUT;
        const unsigned int K = CIN;

        size_t srcMatrixDataSize = 0;
        size_t dstMatrixDataSize = 0;

        OPTIX_CHECK( optixCoopVecMatrixComputeSize(
            optix_context,
            N,
            K,
            layerType[i],
            srcMatrixLayout,
            optRowColumnStrideInBytes,
            &srcMatrixDataSize
            ) );

        OPTIX_CHECK( optixCoopVecMatrixComputeSize(
            optix_context,
            N,
            K,
            layerType[i],
            dstMatrixLayout,
            optRowColumnStrideInBytes,
            &dstMatrixDataSize
            ) );

        OptixCoopVecMatrixDescription& srcLayer = srcLayerDesc[i];
        OptixCoopVecMatrixDescription& dstLayer = dstLayerDesc[i];
        srcLayer.N = dstLayer.N = N;
        srcLayer.K = dstLayer.K = K;
        srcLayer.offsetInBytes  = inferenceData.constants.networkWeightOffsets[i];
        dstLayer.offsetInBytes  = i == 0 ? 0 : dstLayerDesc[i - 1].offsetInBytes + dstLayerDesc[i - 1].sizeInBytes;
        srcLayer.elementType    = layerType[i];
        dstLayer.elementType    = layerType[i];
        srcLayer.layout         = srcMatrixLayout;
        dstLayer.layout         = dstMatrixLayout;
        srcLayer.rowColumnStrideInBytes = optRowColumnStrideInBytes;
        dstLayer.rowColumnStrideInBytes = optRowColumnStrideInBytes;
        srcLayer.sizeInBytes            = static_cast<unsigned int>( srcMatrixDataSize );
        dstLayer.sizeInBytes            = static_cast<unsigned int>( dstMatrixDataSize );
    }

    CUdeviceptr d_srcMatrix;
    CUDA_CHECK( cudaMalloc( (void**)&d_srcMatrix, h_weights.size ) );
    CUDA_CHECK( cudaMemcpy( (void*)d_srcMatrix, h_weights.data, h_weights.size, cudaMemcpyHostToDevice ) );

    OptixNetworkDescription inputNetworkDescription = { srcLayerDesc.data(), static_cast<unsigned int>( srcLayerDesc.size() ) };
    OptixNetworkDescription outputNetworkDescription = { dstLayerDesc.data(), static_cast<unsigned int>( dstLayerDesc.size() ) };

    size_t dst_mats_size = dstLayerDesc.back().offsetInBytes + dstLayerDesc.back().sizeInBytes;  // trick to sum all dstLayer sizes
    size_t src_mats_size = srcLayerDesc.back().offsetInBytes + srcLayerDesc.back().sizeInBytes;  // trick to sum all srcLayer sizes
    size_t src_other_stuff_size = h_weights.size - src_mats_size;
    size_t dst_total_size       = dst_mats_size + src_other_stuff_size;

    CUdeviceptr d_dstMatrix;
    CUDA_CHECK( cudaMalloc( (void**)&d_dstMatrix, dst_total_size ) );

    std::cerr << "NTC weights mem usage: " << dst_total_size << " bytes" << std::endl;

    const int numNetworks = 1;
    const int stride = 0;
    OPTIX_CHECK( optixCoopVecMatrixConvert(
        optix_context,
        CUstream{0},
        numNetworks,
        &inputNetworkDescription,
        d_srcMatrix,
        stride,
        &outputNetworkDescription,
        d_dstMatrix,
        stride) );

    // copy the other stuff after the mats arrays from src to dest
    cudaMemcpy( (void*)(d_dstMatrix + dst_mats_size), (void*)(d_srcMatrix + src_mats_size), src_other_stuff_size, cudaMemcpyDeviceToDevice );

    return (uint8_t*)d_dstMatrix;
}


NtcTextureSet loadNtcFile( ntc::IContext* context, const char* compressedFileName, OptixDeviceContext optixContext, bool enableFP8 )
{
    NtcTextureSet nts;

    // Read the metadata of the ntc file
    ntc::FileStreamWrapper inputStream( context );
    ntc::StreamRange       latentsStreamRange;

    checkNtcStatus( context->OpenFile( compressedFileName, false, inputStream.ptr() ) );
    checkNtcStatus( context->CreateTextureSetMetadataFromStream( inputStream, &nts.metadata ) );
    checkNtcStatus( nts.metadata->GetStreamRangeForLatents( 0, 1, latentsStreamRange ) );

    // Load the whole file into a host buffer
    std::ifstream     file( compressedFileName, std::ios::binary );
    std::vector<char> data;
    file.unsetf( std::ios::skipws );
    std::copy( std::istream_iterator<char>( file ), std::istream_iterator<char>(), std::back_inserter( data ) );
    uint64_t dataSize = data.size();

    std::cerr << "NTC latents mem usage: " << dataSize << " bytes" << std::endl;

    // Copy NTC latents to GPU
    CUDA_CHECK( cudaMalloc( &nts.d_latents, dataSize ) );
    CUDA_CHECK( cudaMemcpy( nts.d_latents, &data[0], dataSize, cudaMemcpyHostToDevice ) );

    ntc::InferenceWeightType weightType = ntc::InferenceWeightType::GenericInt8;
    if( enableFP8 )
        weightType = ntc::InferenceWeightType::GenericFP8;

    // Get the inference data
    ntc::InferenceData* inferenceData = (ntc::InferenceData*)&nts.constants;
    checkNtcStatus( g_ntcContext->MakeInferenceData( nts.metadata, ntc::EntireStream, weightType, inferenceData ) );

    Span weights = getWeightsData( compressedFileName, nts.metadata, weightType );
    if( weights.size > 0 )
        nts.d_mlpWeights = convertWeights( nts.metadata, *inferenceData, enableFP8, weights, optixContext );

    //nts.d_latents      = (uint8_t*)latentsDevicePtr;
    nts.totalChannels  = nts.metadata->GetDesc().channels;
    nts.networkVersion = nts.metadata->GetNetworkVersion();

    // Expecting the following channels in nts:
    // 0..3: base color
    // 4..6: emissive
    // 7..9: normal
    // 10,11,12: occlusion, roughness, metal
    assert( nts.totalChannels >= 13 );

    // Include info on different textures in the texture set.
    nts.numSubTextures = nts.metadata->GetTextureCount();

    // Copy NTC texture metadata to GPU
    NtcTextureSet* d_nts = nullptr;
    CUDA_CHECK( cudaMalloc( &d_nts, sizeof(NtcTextureSet)) );
    CUDA_CHECK( cudaMemcpy( d_nts, &nts, sizeof(NtcTextureSet), cudaMemcpyHostToDevice ) );
    g_params.textureSet = d_nts;

    return nts;
}


//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

void resetView()
{
    g_params.subframe_index = 0;
}


static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    double xpos, ypos;
    glfwGetCursorPos( window, &xpos, &ypos );

    if( action == GLFW_PRESS )
    {
        mouse_button = button;
        trackball.startTracking( static_cast<int>( xpos ), static_cast<int>( ypos ) );
    }
    else
    {
        mouse_button = -1;
    }
}


static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    if( mouse_button == GLFW_MOUSE_BUTTON_LEFT )
    {
        trackball.setViewMode( sutil::Trackball::LookAtFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
    else if( mouse_button == GLFW_MOUSE_BUTTON_RIGHT )
    {
        trackball.setViewMode( sutil::Trackball::EyeFixed );
        trackball.updateTracking( static_cast<int>( xpos ), static_cast<int>( ypos ), width, height );
        camera_changed = true;
    }
}


static void windowSizeCallback( GLFWwindow* window, int32_t res_x, int32_t res_y )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize( res_x, res_y );

    width          = res_x;
    height         = res_y;
    camera_changed = true;
    resize_dirty   = true;
}


static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
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

    g_params.subframe_index = 0;
}


static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    if( trackball.wheelEvent( (int)yscroll ) )
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
//
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --fp8                       Use FP8 inference kernel (default " << (g_params.bound.enableFP8 ? "true" : "false") << ")\n";
    std::cerr << "         --int8                      Use INT8 inference kernel (default " << (g_params.bound.enableFP8 ? "false" : "true") << ")\n";
    std::cerr << "         --maxRegCount | -mrc <int>  Set max register count (for perf tuning, default " << g_max_reg_count << ")\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to " << width << "x" << height << "\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default " << g_params.bound.samples_per_launch << ")\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 0 );
}


void initLaunchParams( const sutil::Scene& scene )
{
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &g_params.accum_buffer ), width * height * sizeof( float4 ) ) );
    g_params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    g_params.subframe_index = 0u;

    const float loffset = scene.aabb().maxExtent();

    std::vector<Light> lights( 2 );
    lights[0].type            = Light::Type::POINT;
    lights[0].point.color     = { 1.0f, 1.0f, 0.8f };
    lights[0].point.intensity = 5.0f;
    lights[0].point.position  = scene.aabb().center() + make_float3( loffset );
    lights[0].point.falloff   = Light::Falloff::QUADRATIC;
    lights[1].type            = Light::Type::POINT;
    lights[1].point.color     = { 0.8f, 0.8f, 1.0f };
    lights[1].point.intensity = 3.0f;
    lights[1].point.position  = scene.aabb().center() + make_float3( -loffset, 0.5f * loffset, -0.5f * loffset );
    lights[1].point.falloff   = Light::Falloff::QUADRATIC;

    g_params.lights.count = static_cast<uint32_t>( lights.size() );
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &g_params.lights.data ), lights.size() * sizeof( Light ) ) );
    CUDA_CHECK( cudaMemcpy(
        reinterpret_cast<void*>( g_params.lights.data ),
        lights.data(),
        lights.size() * sizeof( Light ),
        cudaMemcpyHostToDevice
        ) );

    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_params ), sizeof( LaunchParams ) ) );

    g_params.handle = scene.traversableHandle();
}


void handleCameraUpdate( LaunchParams& params )
{
    if( !camera_changed )
        return;
    camera_changed = false;

    camera.setAspectRatio( static_cast<float>( width ) / static_cast<float>( height ) );
    params.eye = camera.eye();
    camera.UVWFrame( params.U, params.V, params.W );
}


void handleResize( sutil::CUDAOutputBuffer<uchar4>& output_buffer )
{
    if( !resize_dirty )
        return;
    resize_dirty = false;

    output_buffer.resize( width, height );

    // Realloc accumulation buffer
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( g_params.accum_buffer ) ) );
    CUDA_CHECK( cudaMalloc(
                reinterpret_cast<void**>( &g_params.accum_buffer ),
                width*height*sizeof(float4)
                ) );
}


void updateState( sutil::CUDAOutputBuffer<uchar4>& output_buffer, LaunchParams& params )
{
    // Update params on device
    if( camera_changed || resize_dirty )
        params.subframe_index = 0;

    handleCameraUpdate( params );
    handleResize( output_buffer );
}


void launchSubframe( sutil::CUDAOutputBuffer<uchar4>& output_buffer, OptixPipeline& pipeline, OptixShaderBindingTable& sbt )
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    g_params.frame_buffer      = result_buffer_data;
    CUDA_CHECK( cudaMemcpyAsync( reinterpret_cast<void*>( d_params ),
                &g_params,
                sizeof( LaunchParams ),
                cudaMemcpyHostToDevice,
                0 // stream
                ) );

    OPTIX_CHECK( optixLaunch(
                pipeline,
                0,             // stream
                reinterpret_cast<CUdeviceptr>( d_params ),
                sizeof( LaunchParams ),
                &sbt,
                width,  // launch width
                height, // launch height
                1       // launch depth
                ) );
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}


void displaySubframe(
        sutil::CUDAOutputBuffer<uchar4>&  output_buffer,
        sutil::GLDisplay&                 gl_display,
        GLFWwindow*                       window )
{
    // Display
    int framebuf_res_x = 0;   // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;   //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display(
            output_buffer.width(),
            output_buffer.height(),
            framebuf_res_x,
            framebuf_res_y,
            output_buffer.getPBO()
            );
}


void initCameraState( const sutil::Scene& scene )
{
    camera         = scene.camera();
    camera_changed = true;

    trackball.setCamera( &camera );
    trackball.setMoveSpeed( 10.0f );
    trackball.setReferenceFrame( make_float3( 1.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 1.0f ), make_float3( 0.0f, 1.0f, 0.0f ) );
    trackball.setGimbalLock( true );
}


void cleanup(OptixPipeline& pipeline, OptixModule& module, OptixShaderBindingTable& sbt)
{
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( g_params.textureSet ) ) );

    OPTIX_CHECK( optixPipelineDestroy( pipeline ) );
    OPTIX_CHECK( optixProgramGroupDestroy( g_raygen_prog_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( g_radiance_miss_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( g_occlusion_miss_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( g_radiance_hit_group ) );
    OPTIX_CHECK( optixProgramGroupDestroy( g_occlusion_hit_group ) );

    OPTIX_CHECK( optixModuleDestroy( module ) );

    // device context is destroyed in the sutil scene destructor

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( g_sbt.raygenRecord) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( g_sbt.missRecordBase ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( g_sbt.hitgroupRecordBase ) ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_params ) ) );
}


OptixModule createPTXModule( OptixDeviceContext& context, OptixPipelineCompileOptions& pipeline_compile_options )
{
    pipeline_compile_options                       = {};
    pipeline_compile_options.usesMotionBlur        = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compile_options.numPayloadValues      = NUM_PAYLOAD_VALUES;
    pipeline_compile_options.numAttributeValues    = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t input_size = 0;
    const char* input = sutil::getInputData( OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixNeuralTextureKernel.cu", input_size );

    // Use the boundValues struct for our module specialization,
    // i.e., everything in boundValues should be compiled as const
    OptixModuleCompileBoundValueEntry bve;

    bve.pipelineParamOffsetInBytes = offsetof( LaunchParams, bound );

    bve.sizeInBytes   = sizeof( BoundValues );
    bve.boundValuePtr = &g_params.bound;
    bve.annotation    = "bound values";


    OptixModuleCompileOptions module_compile_options = {};
#if OPTIX_DEBUG_DEVICE_CODE
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    module_compile_options.boundValues    = &bve;
    module_compile_options.numBoundValues = 1;

    // For cooperative vectors, increasing the register count improves performance, by avoiding excessive spilling
    // To check for spills, see optix compiler output, and/or inspect SASS and look for preamble stores and a matching set of postamble loads
    // On RTX 6000 Ada, max reg count of 168 seems to be highest perf - best balance of spill vs occupancy
    module_compile_options.maxRegisterCount = g_max_reg_count;

    OptixModule module = 0;
    OPTIX_CHECK_LOG( optixModuleCreate(
                context,
                &module_compile_options,
                &pipeline_compile_options,
                input,
                input_size,
                LOG, &LOG_SIZE,
                &module
                ) );
    return module;
}


void createProgramGroups( OptixDeviceContext& context, OptixModule module )
{
    OptixProgramGroupOptions program_group_options = {};

    //
    // Ray generation
    //
    {
        OptixProgramGroupDesc raygen_prog_group_desc    = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__pinhole";

        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            LOG,
            &LOG_SIZE,
            &g_raygen_prog_group
            ) );
    }

    //
    // Miss
    //
    {
        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__constant_radiance";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,                             // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &g_radiance_miss_group
            ) );

        memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__occlusion";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,                             // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &g_occlusion_miss_group
            ) );
    }

    //
    // Hit group
    //
    {
        OptixProgramGroupDesc hit_prog_group_desc        = {};
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
             context,
             &hit_prog_group_desc,
             1,                             // num program groups
             &program_group_options,
             LOG, &LOG_SIZE,
             &g_radiance_hit_group
             ) );

        memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
             context,
             &hit_prog_group_desc,
             1,                             // num program groups
             &program_group_options,
             LOG, &LOG_SIZE,
             &g_occlusion_hit_group
             ) );
    }
}


OptixPipeline createPipeline(OptixDeviceContext& context, OptixPipelineCompileOptions& pipeline_compile_options)
{
    OptixProgramGroup program_groups[] =
    {
        g_raygen_prog_group,
        g_radiance_miss_group,
        g_occlusion_miss_group,
        g_radiance_hit_group,
        g_occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth            = MAX_TRACE_DEPTH;

    OptixPipeline m_pipeline;
    OPTIX_CHECK_LOG( optixPipelineCreate(
                context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof( program_groups ) / sizeof( program_groups[0] ),
                LOG, &LOG_SIZE,
                &m_pipeline
                ) );
    return m_pipeline;
}


OptixShaderBindingTable createSBT( OptixDeviceContext&                                          context,
                                   const std::vector<std::shared_ptr<sutil::Scene::Instance>>&  instances,
                                   const std::vector<std::shared_ptr<sutil::Scene::MeshGroup>>& meshes,
                                   const std::vector<MaterialData>&                             materials )
{
    OptixShaderBindingTable m_sbt = {};
    {
        const size_t raygen_record_size = sizeof( sutil::EmptyRecord );
        CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_sbt.raygenRecord ), raygen_record_size ) );

        sutil::EmptyRecord rg_sbt;
        OPTIX_CHECK( optixSbtRecordPackHeader( g_raygen_prog_group, &rg_sbt ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( m_sbt.raygenRecord ),
                    &rg_sbt,
                    raygen_record_size,
                    cudaMemcpyHostToDevice
                    ) );
    }

    {
        const size_t miss_record_size = sizeof( sutil::EmptyRecord );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_sbt.missRecordBase ),
                    miss_record_size*RAY_TYPE_COUNT
                    ) );

        sutil::EmptyRecord ms_sbt[ RAY_TYPE_COUNT ];
        OPTIX_CHECK( optixSbtRecordPackHeader( g_radiance_miss_group,  &ms_sbt[0] ) );
        OPTIX_CHECK( optixSbtRecordPackHeader( g_occlusion_miss_group, &ms_sbt[1] ) );

        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( m_sbt.missRecordBase ),
                    ms_sbt,
                    miss_record_size*RAY_TYPE_COUNT,
                    cudaMemcpyHostToDevice
                    ) );
        m_sbt.missRecordStrideInBytes = static_cast<uint32_t>( miss_record_size );
        m_sbt.missRecordCount         = RAY_TYPE_COUNT;
    }

    {
        std::vector<HitGroupRecord> hitgroup_records;
        for( const auto& instance : instances )
        {
            const auto mesh = meshes[instance->mesh_idx];
            for( size_t i = 0; i < mesh->material_idx.size(); ++i )
            {
                HitGroupRecord rec = {};
                OPTIX_CHECK( optixSbtRecordPackHeader( g_radiance_hit_group, &rec ) );
                GeometryData::TriangleMesh triangle_mesh = {};
                triangle_mesh.normals                    = mesh->normals[i];
                triangle_mesh.positions                  = mesh->positions[i];
                for( size_t j = 0; j < GeometryData::num_texcoords; ++j )
                    triangle_mesh.texcoords[j] = mesh->texcoords[j][i];
                triangle_mesh.colors  = mesh->colors[i];
                triangle_mesh.indices = mesh->indices[i];
                rec.data.geometry_data.setTriangleMesh( triangle_mesh );

                const int32_t mat_idx = mesh->material_idx[i];
                if( mat_idx >= 0 )
                    rec.data.material_data = materials[mat_idx];
                else
                    rec.data.material_data = MaterialData();
                hitgroup_records.push_back( rec );

                OPTIX_CHECK( optixSbtRecordPackHeader( g_occlusion_hit_group, &rec ) );
                hitgroup_records.push_back( rec );
            }
        }

        const size_t hitgroup_record_size = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_sbt.hitgroupRecordBase ),
                    hitgroup_record_size*hitgroup_records.size()
                    ) );
        CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( m_sbt.hitgroupRecordBase ),
                    hitgroup_records.data(),
                    hitgroup_record_size*hitgroup_records.size(),
                    cudaMemcpyHostToDevice
                    ) );

        m_sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>( hitgroup_record_size );
        m_sbt.hitgroupRecordCount         = static_cast<unsigned int>( hitgroup_records.size() );
    }

    return m_sbt;
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main( int argc, char* argv[] )
{
    g_params.bound.samples_per_launch = 1;
    g_params.bound.enableFP8          = true;
    g_params.bound.miss_color         = make_float3( 0.1f );

    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

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
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
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
            sutil::parseDimensions( dims_arg.c_str(), width, height );
        }
        else if( arg == "--launch-samples" || arg == "-s" )
        {
            if( i >= argc - 1 )
                printUsageAndExit( argv[0] );
            g_params.bound.samples_per_launch = atoi( argv[++i] );
        }
        else if( arg == "--fp8" )
        {
            g_params.bound.enableFP8 = true;
        }
        else if( arg == "--int8" )
        {
            g_params.bound.enableFP8 = false;
        }
        else if( arg == "-mrc" || arg == "--maxRegCount" )
        {
            g_max_reg_count = atoi( argv[++i] );
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit( argv[0] );
        }
    }

    std::string infile      = sutil::sampleDataFilePath( "NeuralTexture/WaterBottle_noTextures.gltf" );
    std::string textureFile = sutil::sampleDataFilePath( "NeuralTexture/WaterBottle.ntc" );

    try
    {
        sutil::Scene scene;
        sutil::loadScene( infile, scene );
        scene.finalize(/*create_pipeline=*/ false, 2);

        OptixDeviceContext context = scene.context();

        OPTIX_CHECK( optixInit() );  // Need to initialize function table.
        initCameraState( scene );
        initLaunchParams( scene );

        g_ntcContext = makeNtcContext();

        //
        // Load the NTC Texture onto host & device
        //
        NtcTextureSet nts = loadNtcFile( g_ntcContext, textureFile.c_str(), context, g_params.bound.enableFP8 );

        g_params.bound.networkVersion = nts.networkVersion;

        //
        // Initialize our pipeline & SBT, since we didn't use the scene class built-in pipeline/SBT
        // 
        OptixPipelineCompileOptions pipeline_compile_options = {};

        const std::vector<std::shared_ptr<sutil::Scene::Instance>>&  instances = scene.instances();
        const std::vector<std::shared_ptr<sutil::Scene::MeshGroup>>& meshes    = scene.meshes();
        const std::vector<MaterialData>&                             materials = scene.materials();

        OptixModule module = createPTXModule( context, pipeline_compile_options );
        createProgramGroups( context, module );

        OptixPipeline           pipeline = createPipeline( context, pipeline_compile_options );
        OptixShaderBindingTable sbt      = createSBT( context, instances, meshes, materials );

        if( outfile.empty() )
        {
            GLFWwindow* window = sutil::initUI( "optixNeuralTexture", width, height );
            glfwSetMouseButtonCallback  ( window, mouseButtonCallback   );
            glfwSetCursorPosCallback    ( window, cursorPosCallback     );
            glfwSetWindowSizeCallback   ( window, windowSizeCallback    );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback          ( window, keyCallback           );
            glfwSetScrollCallback       ( window, scrollCallback        );
            glfwSetWindowUserPointer    ( window, &g_params             );

            resetView();

            //
            // Render loop
            //
            {
                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, width, height );
                sutil::GLDisplay                gl_display;

                std::chrono::duration<double> state_update_time( 0.0 );
                std::chrono::duration<double> render_time( 0.0 );
                std::chrono::duration<double> display_time( 0.0 );

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();

                    updateState( output_buffer, g_params );
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe( output_buffer, pipeline, sbt );
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe( output_buffer, gl_display, window );
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats( state_update_time, render_time, display_time );

                    glfwSwapBuffers( window );

                    ++g_params.subframe_index;
                } while( !glfwWindowShouldClose( window ) );
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI( window );
        }
        else
        {
            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }

            {
                // this scope is for output_buffer, to ensure the destructor is called bfore glfwTerminate()

                sutil::CUDAOutputBuffer<uchar4> output_buffer( output_buffer_type, width, height );
                handleCameraUpdate( g_params );
                handleResize( output_buffer );
                launchSubframe( output_buffer, pipeline, sbt );

                sutil::ImageBuffer buffer;
                buffer.data         = output_buffer.getHostPointer();
                buffer.width        = output_buffer.width();
                buffer.height       = output_buffer.height();
                buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

                sutil::saveImage( outfile.c_str(), buffer, false );
            }

            if( output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP )
            {
                glfwTerminate();
            }
        }

        cleanup(pipeline, module, sbt);
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
