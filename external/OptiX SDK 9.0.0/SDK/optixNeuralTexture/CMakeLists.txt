#
# SPDX-FileCopyrightText: Copyright (c) 2019 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


if( CUDA_NVRTC_ENABLED )
  message(WARNING "optixNeuralTexture is not expected to compile with nvrtc. To build this sample, please disable nvrtc and use nvcc instead")
  return()
endif()


# libntc enforces cmake 3.18 - skip this sample gracefully if your cmake is older
if (CMAKE_VERSION VERSION_LESS 3.18)
  message(WARNING "CMake version 3.18 or greater is needed for optixNeuralTexture. Skipping.")
  return()
endif()


# libntc requires c++17
set( CMAKE_CXX_STANDARD 17 )


#-----------------
# Build NTC library with the following options

set( NTC_WITH_CUDA OFF CACHE BOOL "" )
set( NTC_WITH_DX12 OFF CACHE BOOL "" )
set( NTC_WITH_VULKAN OFF CACHE BOOL "" )
set( NTC_BUILD_SHARED FALSE )

# libntc code is currently embedded in this sample, as it is not public at the time of writing.
# In the future, this sample may link to and automatically download the public version of libntc.
# Find the latest version of the NTC SDK on Github, including NTC explorer and DX, Vulkan, CUDA, and Slang examples.
add_subdirectory( libntc )

#-----------------

set_target_properties(libntc PROPERTIES COMPILE_WARNING_AS_ERROR FALSE)
target_compile_options(libntc PRIVATE -w)

include_directories(${libntc_SOURCE_DIR}/include)

# nuke all the arch flags here (so we can override)
list(FIND SAMPLES_NVCC_FLAGS "-arch" index )
while( index GREATER -1 )
  list(REMOVE_ITEM SAMPLES_NVCC_FLAGS "-arch")
  list(REMOVE_AT SAMPLES_NVCC_FLAGS ${index})
  list(FIND SAMPLES_NVCC_FLAGS "-arch" index )
endwhile()

OPTIX_add_sample_executable( optixNeuralTexture target_name
  ${SAMPLES_CUDA_DIR}/helpers.h
  NtcInference.h
  NtcTexture.h
  NtcTextureSet.h
  optixNeuralTexture.cpp
  optixNeuralTextureKernel.cu
  optixNeuralTexture.h
  optixNeuralTextureUtil.h
  OPTIONS -arch sm_60 -rdc true --std=c++17
  )

target_link_libraries( ${target_name}
  ${CUDA_LIBRARIES}
  libntc
  )
