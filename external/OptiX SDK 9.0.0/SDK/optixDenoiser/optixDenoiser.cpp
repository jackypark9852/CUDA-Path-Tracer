/*

 * SPDX-FileCopyrightText: Copyright (c) 2020 - 2024  NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include "OptiXDenoiser.h"

#include <sutil/Exception.h>
#include <sutil/sutil.h>

#define TINYEXR_IMPLEMENTATION
#include <tinyexr/tinyexr.h>

#include <stdlib.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>


//------------------------------------------------------------------------------
//
//  optixDenoiser -- Demonstration of the OptiX denoising API.
//
//------------------------------------------------------------------------------

void printUsageAndExit( const std::string& argv0 )
{
    std::cout << "Usage  : " << argv0 << " [options] {-A | --AOV aov.exr} color.exr\n"
              << "Options: -n | --normal <normal.exr | layer name>\n"
              << "         -a | --albedo <albedo.exr | layer name>\n"
              << "         -b | --beauty <layer name>\n"
              << "         -f | --flow   <flow.exr | layer name>\n"
              << "         -A | --AOV    <aov.exr | layer name>\n"
              << "         -S            <specular aov.exr | layer name>\n"
              << "         -T            <flowTrustworthiness.exr | layer name>\n"
              << "         -o | --out    <out.exr> Defaults to 'denoised.exr'\n"
              << "         -F | --Frames <int-int> first-last frame number in sequence\n"
              << "         -e | --exposure <float> apply exposure on output images\n"
              << "         -t | --tilesize <int> <int> use tiling to save GPU memory\n"
              << "         -alpha denoise alpha channel\n"
              << "         -fmul <x y> multiply flow vector components\n"
              << "         -fp32 write images with 32-bit precision, default 16 bit\n"
              << "         -up2 upscale image by factor of 2\n"
              << "         -z apply flow to input images (no denoising), for flow vector verification\n"
              << "in sequences, first occurrence of '+' characters substring in filenames is replaced by framenumber\n"
              << std::endl;
    exit( 0 );
}

static void freeImageBuffer( sutil::ImageBuffer& image )
{
    switch( image.pixel_format )
    {
        case sutil::FLOAT4:
            delete[] reinterpret_cast<float4*>( image.data );
            break;
        case sutil::FLOAT3:
            delete[] reinterpret_cast<float3*>( image.data );
            break;
        default:
            break;
    }
    image.data   = nullptr;
    image.width  = 0;
    image.height = 0;
}

static double getCurrentTime()
{
    return std::chrono::duration_cast< std::chrono::duration< double > >
        ( std::chrono::high_resolution_clock::now().time_since_epoch() ).count();
}

// Layers given on the command line (-a, -n, -f, -b) are passed as 'layerName'. If 'layerName' is 0,
// layers are not searched and RGB is assumed.
// An input filename (possibly containing all layers) on the command line is passed as 'inputFileName'.
// If the inputFileName has a layer with the name 'layerName', it is loaded from 'inputFileName',
// otherwise it is assumed that 'layerName' is the filename to load from (layer name RGB).
// Note that we use a modified tinyexr.h file which has an alternative component search for XYZ,
// as this is used for normals and velocity. The original version searches only for RGB.

static sutil::ImageBuffer loadImageLayer( const char* inputFileName, const char* layerName, EXRImage* cachedImage )
{
    const char* err = nullptr;
    bool layerExists = false;

    std::string baseErrorMessage = std::string( "failed to load from file \"" ) + std::string( inputFileName ) + "\" ";

    if( layerName && std::string( layerName ).find( "exr" ) == std::string::npos )
    {
        const char** layerNames = nullptr;
        int numLayers = 0;

        int32_t res = EXRLayers( inputFileName, &layerNames, &numLayers, &err);
        if( res != TINYEXR_SUCCESS )
        {
            if( err )
            {
                sutil::Exception e( ( baseErrorMessage + err ).c_str() );
                FreeEXRErrorMessage( err );
                throw e;
            }
            else
            {
                throw( sutil::Exception( baseErrorMessage.c_str() ) );
            }
        }
        for( int i=0; i < numLayers; i++ )
        {
            if( std::string( layerNames[i] ) == std::string( layerName ) )
                layerExists = true;
        }
        if( !layerExists )
        {
            std::string estr = baseErrorMessage +
                               std::string( ", layer \"" ) + std::string( layerName ) + std::string( "\" not found.\n" );
            sutil::Exception e( estr.c_str() );
            throw e;
        }
    }

    float*  data = nullptr;
    int32_t res, w, h;
    if( layerExists )
        res = LoadEXRWithLayer( &data, &w, &h, inputFileName, layerName, &err, cachedImage );
    else	// load RGB from input file (if layerName is given, it must be an EXR file to load).
                // if layerName is null, it must be beauty, so load RGB from beauty EXR
        res = LoadEXR( &data, &w, &h, inputFileName, &err );

    if( res != TINYEXR_SUCCESS)
    {
        if ( err )
        {
            sutil::Exception e( ( baseErrorMessage + err ).c_str() );
            FreeEXRErrorMessage( err );
            throw e;
        }
        else
        {
            throw sutil::Exception( baseErrorMessage.c_str() );
        }
    }

    sutil::ImageBuffer image;
    image.width  = w;
    image.height = h;
    image.data   = new float4[ image.width*image.height ];
    image.pixel_format = sutil::FLOAT4;
    memcpy( image.data, data, sizeof(float) * 4 * w * h );

    free( data );

    return image;
}

// save image to EXR file

static void saveImageEXR( const char* fname, const sutil::ImageBuffer& image, bool fp16 )
{
    const std::string filename( fname );

    switch( image.pixel_format )
    {
        case sutil::BufferImageFormat::FLOAT3:
        {
            const char* err;
            int32_t ret = SaveEXR(
                    reinterpret_cast<float*>( image.data ),
                    image.width,
                    image.height,
                    3, // num components
                    static_cast<int32_t>( fp16 ), // save_as_fp16
                    filename.c_str(),
                    &err );

            if( ret != TINYEXR_SUCCESS )
                throw sutil::Exception( ( "error saving image: " + std::string( err ) ).c_str() );

        } break;

        case sutil::BufferImageFormat::FLOAT4:
        {
            const char* err;
            int32_t ret = SaveEXR(
                    reinterpret_cast<float*>( image.data ),
                    image.width,
                    image.height,
                    4, // num components
                    static_cast<int32_t>( fp16 ), // save_as_fp16
                    filename.c_str(),
                    &err );

            if( ret != TINYEXR_SUCCESS )
                throw sutil::Exception( ( "error saving image: " + std::string( err ) ).c_str() );
        } break;

        default:
        {
            throw sutil::Exception( "error saving image: Unrecognized image buffer pixel format.\n" );
        }
    }
}

// filename is copied to result and the first sequence of "+" characters is
// replaced (using leading zeros) with framename.
// true is returned if the framenumber is -1 or if the function was successful.

static bool getFrameFilename( std::string& result, const std::string& filename, const std::string& colorFilename, int frame )
{
    if( filename.find( "exr" ) == std::string::npos )   // no ext extension, it must be a layer name, search in colorFilename
    {                                                   // which already has the name for the given frame
        result = colorFilename;
        return true;
    }

    result = filename;
    if( frame == -1 )
        return true;
    size_t nplus = 0;
    size_t ppos  = result.find( '+' );
    if( ppos == std::string::npos )
        return true;  // static filename without "+" characters
    size_t cpos = ppos;
    while( result[cpos] != 0 && result[cpos] == '+' )
    {
        nplus++;
        cpos++;
    }
    std::string fn = std::to_string( frame );
    if( fn.length() > nplus )
    {
        std::cout << "illegal temporal filename, framenumber requires " << fn.length()
                  << " digits, \"+\" placeholder length: " << nplus << "too small" << std::endl;
        return false;
    }
    for( size_t i = 0; i < nplus; i++ )
        result[ppos + i] = '0';
    for( size_t i = 0; i < fn.length(); i++ )
        result[ppos + nplus - 1 - i] = fn[fn.length() - 1 - i];
    return true;
}

int32_t main( int32_t argc, char** argv )
{
    if( argc < 2 )
        printUsageAndExit( argv[0] );

    std::string color_filename = argv[argc - 1];

    std::string              beauty_filename;
    std::string              normal_filename;
    std::string              albedo_filename;
    std::string              flow_filename;
    std::string              flowtrust_filename;
    std::string              output_filename = "denoised.exr";
    std::vector<std::string> aov_filenames;
    bool                     kpMode     = true;
    bool                     applyFlow  = false;
    float                    exposure   = 0.f;
    int                      firstFrame = -1, lastFrame = -1;
    unsigned int             tileWidth = 0, tileHeight = 0;
    bool                     upscale2x = false;
    OptixDenoiserAlphaMode   alphaMode = OPTIX_DENOISER_ALPHA_MODE_COPY;
    bool                     specularMode = 0;
    float                    xmul = 1.f, ymul = 1.f;
    bool                     multiplyFlow = false;
    bool                     writeFP16 = true;

    for( int32_t i = 1; i < argc - 1; ++i )
    {
        std::string arg( argv[i] );

        if( arg == "-b" || arg == "--beauty" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            beauty_filename = argv[++i];
        }
        else if( arg == "-n" || arg == "--normal" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            normal_filename = argv[++i];
        }
        else if( arg == "-a" || arg == "--albedo" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            albedo_filename = argv[++i];
        }
        else if( arg == "-e" || arg == "--exposure" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            exposure = std::stof( argv[++i] );
        }
        else if( arg == "-f" || arg == "--flow" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            flow_filename = argv[++i];
        }
        else if( arg == "-T" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            flowtrust_filename = argv[++i];
        }
        else if( arg == "-o" || arg == "--out" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            output_filename = argv[++i];
        }
        else if( arg == "-t" || arg == "--tilesize" )
        {
            if( i == argc - 3 )
                printUsageAndExit( argv[0] );
            tileWidth  = atoi( argv[++i] );
            tileHeight = atoi( argv[++i] );
        }
        else if( arg == "-A" || arg == "--AOV" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            aov_filenames.push_back( std::string( argv[++i] ) );
        }
        else if( arg == "-S" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            aov_filenames.push_back( std::string( argv[++i] ) );
            specularMode = true;
        }
        else if( arg == "-k" )
        {
            kpMode = true;
        }
        else if( arg == "-d" )
        {
            kpMode = false;
        }
        else if( arg == "-fp32" )
        {
            writeFP16 = false;
        }
        else if( arg == "-z" )
        {
            applyFlow = true;
        }
        else if( arg == "-up2" )
        {
            upscale2x = true;
        }
        else if( arg == "-alpha" )
        {
            alphaMode = OPTIX_DENOISER_ALPHA_MODE_DENOISE;
        }
        else if( arg == "-F" || arg == "--Frames" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            std::string s( argv[++i] );
            size_t      cpos = s.find( '-' );
            if( cpos == 0 || cpos == s.length() - 1 || cpos == std::string::npos )
                printUsageAndExit( argv[0] );
            firstFrame = atoi( s.substr( 0, cpos ).c_str() );
            lastFrame  = atoi( s.substr( cpos + 1 ).c_str() );
            if( firstFrame < 0 || lastFrame < 0 || firstFrame > lastFrame )
            {
                std::cout << "illegal frame range, first frame must be <= last frame and >= 0" << std::endl;
                exit( 0 );
            }
        }
        else if( arg == "-fmul" )
        {
            if( i == argc - 2 )
                printUsageAndExit( argv[0] );
            std::string s1( argv[++i] );
            xmul = float( atof( s1.c_str() ) );
            std::string s2( argv[++i] );
            ymul = float( atof( s2.c_str() ) );
            multiplyFlow = true;
        }
        else
        {
            printUsageAndExit( argv[0] );
        }
    }

    bool temporalMode = bool( firstFrame != -1 );

    if( temporalMode && flow_filename.empty() )
    {
        std::cout << "temporal mode enabled, flow filename not specified" << std::endl;
        exit( 0 );
    }

    sutil::ImageBuffer              color     = {};
    sutil::ImageBuffer              normal    = {};
    sutil::ImageBuffer              albedo    = {};
    sutil::ImageBuffer              flow      = {};
    sutil::ImageBuffer              flowtrust = {};

    unsigned int outScale = upscale2x ? 2 : 1;

    try
    {
        EXRImage exrImage = {};

        OptiXDenoiser denoiser;
        for( int frame = firstFrame; frame <= lastFrame; frame++ )
        {
            std::vector<sutil::ImageBuffer> aovs;

            const double t0 = getCurrentTime();
            std::cout << "Loading inputs ";
            if( frame != -1 )
                std::cout << "for frame " << frame;
            std::cout << " ..." << std::endl;

            std::string frame_filename;
            if( !getFrameFilename( frame_filename, color_filename, std::string(""), frame ) )
            {
                std::cout << "cannot open color file" << std::endl;
                exit( 0 );
            }
            std::string frame_filename_input = frame_filename;

            color = loadImageLayer( frame_filename.c_str(), beauty_filename.empty() ? 0 : beauty_filename.c_str(), &exrImage );
            std::cout << "\tLoaded color image " << frame_filename << " (" << color.width << "x" << color.height << ")"
                      << std::endl;

            if( !normal_filename.empty() )
            {
                if( !getFrameFilename( frame_filename, normal_filename, frame_filename_input, frame ) )
                {
                    std::cout << "cannot open normal file" << std::endl;
                    exit( 0 );
                }

                // allocate four channels. only two/three channels used depending on model.
                normal = loadImageLayer( frame_filename.c_str(), normal_filename.c_str(), &exrImage );
                std::cout << "\tLoaded normal image " << frame_filename << std::endl;
            }

            if( !albedo_filename.empty() )
            {
                if( !getFrameFilename( frame_filename, albedo_filename, frame_filename_input, frame ) )
                {
                    std::cout << "cannot open albedo file" << std::endl;
                    exit( 0 );
                }
                // allocate four channels. only three channels used.
                albedo = loadImageLayer( frame_filename.c_str(), albedo_filename.c_str(), &exrImage );
                std::cout << "\tLoaded albedo image " << frame_filename << std::endl;
            }

            if( frame > firstFrame && !flow_filename.empty() )
            {
                if( !getFrameFilename( frame_filename, flow_filename, frame_filename_input, frame ) )
                {
                    std::cout << "cannot open flow file" << std::endl;
                    exit( 0 );
                }
                // allocate four channels. only two channels used.
                // sutil::loadImage handles only 3 and 4 channels.
                flow = loadImageLayer( frame_filename.c_str(), flow_filename.c_str(), &exrImage );
                std::cout << "\tLoaded flow image " << frame_filename << std::endl;

                if( multiplyFlow )
                {
                    float* data = (float*)flow.data;
                    for( unsigned int i=0; i < flow.width * flow.height; i++ )
                    {
                        data[i*4+0] *= xmul;
                        data[i*4+1] *= ymul;
                    }
                }
            }

            if( !flowtrust_filename.empty() )
            {
                if( !getFrameFilename( frame_filename, flowtrust_filename, frame_filename_input, frame ) )
                {
                    std::cout << "cannot open flowTrustworthiness file" << std::endl;
                    exit( 0 );
                }
                // allocate four channels. only three channels used.
                flowtrust = loadImageLayer( frame_filename.c_str(), flowtrust_filename.c_str(), &exrImage );
                std::cout << "\tLoaded flowTrustworthiness image " << frame_filename << std::endl;
            }

            for( size_t i = 0; i < aov_filenames.size(); i++ )
            {
                if( !getFrameFilename( frame_filename, aov_filenames[i], frame_filename_input, frame ) )
                {
                    std::cout << "cannot open aov file" << std::endl;
                    exit( 0 );
                }
                aovs.push_back( loadImageLayer( frame_filename.c_str(), aov_filenames[i].c_str(), &exrImage ) );
                std::cout << "\tLoaded aov image " << frame_filename << std::endl;
            }

            const double t1 = getCurrentTime();
            std::cout << "\tLoad inputs from disk     :" << std::fixed << std::setw( 8 ) << std::setprecision( 2 )
                      << ( t1 - t0 ) * 1000.0 << " ms" << std::endl;

            SUTIL_ASSERT( color.pixel_format == sutil::FLOAT4 );
            SUTIL_ASSERT( !albedo.data || albedo.pixel_format == sutil::FLOAT4 );
            SUTIL_ASSERT( !normal.data || normal.pixel_format == sutil::FLOAT4 );
            SUTIL_ASSERT( !flow.data || flow.pixel_format == sutil::FLOAT4 );
            for( size_t i = 0; i < aov_filenames.size(); i++ )
                SUTIL_ASSERT( aovs[i].pixel_format == sutil::FLOAT4 );

            OptiXDenoiser::Data data;
            data.width     = color.width;
            data.height    = color.height;
            data.color     = reinterpret_cast<float*>( color.data );
            data.albedo    = reinterpret_cast<float*>( albedo.data );
            data.normal    = reinterpret_cast<float*>( normal.data );
            data.flow      = reinterpret_cast<float*>( flow.data );
            data.flowtrust = reinterpret_cast<float*>( flowtrust.data );

            // set AOVs
            for( size_t i = 0; i < aovs.size(); i++ )
                data.aovs.push_back( reinterpret_cast<float*>( aovs[i].data ) );

            // allocate outputs
            for( size_t i = 0; i < 1 + aovs.size(); i++ )
                data.outputs.push_back( new float[outScale * color.width * outScale * color.height * 4] );

            std::cout << "Denoising ..." << std::endl;

            if( frame == firstFrame )
            {
                const double t0 = getCurrentTime();
                denoiser.init( data, tileWidth, tileHeight, kpMode, temporalMode, applyFlow, upscale2x, alphaMode, specularMode );
                const double t1 = getCurrentTime();
                std::cout << "\tAPI Initialization        :" << std::fixed << std::setw( 8 ) << std::setprecision( 2 )
                          << ( t1 - t0 ) * 1000.0 << " ms" << std::endl;
            }
            else
            {
                denoiser.update( data );
            }

            {
                const double t0 = getCurrentTime();
                denoiser.exec();
                const double t1 = getCurrentTime();
                std::cout << "\tDenoise frame             :" << std::fixed << std::setw( 8 ) << std::setprecision( 2 )
                          << ( t1 - t0 ) * 1000.0 << " ms" << std::endl;
            }

            {
                const double t0 = getCurrentTime();               
                denoiser.getResults();
                const double t1 = getCurrentTime();
                std::cout << "\tCleanup state/copy to host:" << std::fixed << std::setw( 8 ) << std::setprecision( 2 )
                          << ( t1 - t0 ) * 1000.0 << " ms" << std::endl;
            }

            // AOVs are not written when speclarMode is set. A single specular AOV is expected in this mode,
            // to keep the sample code simple.
            size_t numOut = specularMode ? 1 : 1 + aovs.size();

            {
                const double t0 = getCurrentTime();

                for( size_t i = 0; i < numOut; i++ )
                {
                    sutil::ImageBuffer output_image;
                    output_image.width        = outScale * color.width;
                    output_image.height       = outScale * color.height;
                    output_image.data         = data.outputs[i];
                    output_image.pixel_format = sutil::FLOAT4;

                    frame_filename = output_filename;
                    getFrameFilename( frame_filename, output_filename, std::string(""), frame );
                    if( i > 0 )
                    {
                        std::string basename = aov_filenames[i - 1].substr( aov_filenames[i - 1].find_last_of( "/\\" ) + 1 );
                        std::string::size_type const p( basename.find_last_of( '.' ) );
                        std::string                  b = basename.substr( 0, p );
                        frame_filename.insert( frame_filename.rfind( '.' ), "_" + b + "_denoised" );
                    }
                    if( exposure != 0.f )
                    {
                        for( unsigned int p = 0; p < output_image.width * output_image.height; p++ )
                        {
                            float* f = &( (float*)output_image.data )[p * 4 + 0];
                            f[0] *= std::pow( 2.f, exposure );
                            f[1] *= std::pow( 2.f, exposure );
                            f[2] *= std::pow( 2.f, exposure );
                        }
                    }
                    std::cout << "Saving results to '" << frame_filename << "'..." << std::endl;
                    saveImageEXR( frame_filename.c_str(), output_image, writeFP16 );
                }

                const double t1 = getCurrentTime();
                std::cout << "\tSave output to disk       :" << std::fixed << std::setw( 8 ) << std::setprecision( 2 )
                          << ( t1 - t0 ) * 1000.0 << " ms" << std::endl;
            }

            freeImageBuffer( color );
            freeImageBuffer( albedo );
            freeImageBuffer( normal );
            freeImageBuffer( flow );
            freeImageBuffer( flowtrust );
            for( size_t i = 0; i < aovs.size(); i++ )
                freeImageBuffer( aovs[i] );
            for( size_t i = 0; i < 1 + aovs.size(); i++ )
                delete[]( data.outputs[i] );

            if( exrImage.num_channels > 0 )
            {
                FreeEXRImage( &exrImage );
                exrImage = {};
            }
        }

        denoiser.finish();
    }
    catch( std::exception& e )
    {
        std::cerr << "ERROR: exception caught '" << e.what() << "'" << std::endl;
    }
}
