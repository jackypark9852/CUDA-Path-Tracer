# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

cmake_minimum_required(VERSION 3.18)

set(LANGUAGES CXX)

option(NTC_BUILD_SHARED "Build LibNTC as a shared library (DLL or .so)" ON)
option(NTC_WITH_CUDA "Compile NTC with CUDA support" ON)
option(NTC_WITH_DX12 "Compile NTC with DX12 support" "${WIN32}")
option(NTC_WITH_VULKAN "Compile NTC with Vulkan support" ON)
option(NTC_WITH_PREBUILT_SHADERS "Compile NTC with pre-built shaders for decompression etc." ON)

if (NTC_WITH_CUDA)
    list(APPEND LANGUAGES CUDA)
endif()

project(
    libntc
    DESCRIPTION "Neural Texture Compression Library"
    LANGUAGES ${LANGUAGES}
)

# if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src")
    
# LibNTC source code available
add_subdirectory(external/rapidjson)
add_subdirectory(src)

# else()
#     # LibNTC delivered as a binary build, use an IMPORTED target
#     add_library(libntc SHARED IMPORTED GLOBAL)
#     target_include_directories(libntc INTERFACE include)
#     if (WIN32)
#         set_target_properties(libntc PROPERTIES
#             IMPORTED_LOCATION_DEBUG          "${CMAKE_CURRENT_SOURCE_DIR}/bin/windows-x64/Debug/libntc.dll"
#             IMPORTED_LOCATION_MINSIZEREL     "${CMAKE_CURRENT_SOURCE_DIR}/bin/windows-x64/MinSizeRel/libntc.dll"
#             IMPORTED_LOCATION_RELEASE        "${CMAKE_CURRENT_SOURCE_DIR}/bin/windows-x64/Release/libntc.dll"
#             IMPORTED_LOCATION_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/bin/windows-x64/RelWithDebInfo/libntc.dll"
#             IMPORTED_IMPLIB_DEBUG            "${CMAKE_CURRENT_SOURCE_DIR}/bin/windows-x64/Debug/libntc.lib"
#             IMPORTED_IMPLIB_MINSIZEREL       "${CMAKE_CURRENT_SOURCE_DIR}/bin/windows-x64/MinSizeRel/libntc.lib"
#             IMPORTED_IMPLIB_RELEASE          "${CMAKE_CURRENT_SOURCE_DIR}/bin/windows-x64/Release/libntc.lib"
#             IMPORTED_IMPLIB_RELWITHDEBINFO   "${CMAKE_CURRENT_SOURCE_DIR}/bin/windows-x64/RelWithDebInfo/libntc.lib"
#         )
#     else()
#         set_target_properties(libntc PROPERTIES
#             IMPORTED_LOCATION_DEBUG          "${CMAKE_CURRENT_SOURCE_DIR}/bin/linux-x64/Debug/libntc.so"
#             IMPORTED_LOCATION_MINSIZEREL     "${CMAKE_CURRENT_SOURCE_DIR}/bin/linux-x64/MinSizeRel/libntc.so"
#             IMPORTED_LOCATION_RELEASE        "${CMAKE_CURRENT_SOURCE_DIR}/bin/linux-x64/Release/libntc.so"
#             IMPORTED_LOCATION_RELWITHDEBINFO "${CMAKE_CURRENT_SOURCE_DIR}/bin/linux-x64/RelWithDebInfo/libntc.so"
#         )
#     endif()
# endif()

# # Function to copy libntc's DLL and maybe PDB into the output directory of some other target.
# # Use like this:  libntc_copy_binary(TARGET my_target OUTPUT_DIRECTORY "my_target_output_dir")
# function(libntc_copy_binary)
#     set(options "")
#     set(oneValueArgs TARGET OUTPUT_DIRECTORY)
#     set(multiValueArgs "")
#     cmake_parse_arguments(params "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
#  
#     # Copy the main output (.dll)
#     add_custom_command(TARGET ${params_TARGET} POST_BUILD COMMAND
#         cmake -E copy_if_different "$<TARGET_FILE:libntc>" "${params_OUTPUT_DIRECTORY}/")
#  
#     get_target_property(libntc_imported libntc IMPORTED)
#     
#     # Copy the .pdb file - only on Windows and only on configs that create .pdb's
#     if(NOT libntc_imported AND WIN32)
#         # Prepare the copy command 
#         set(copy_command_base cmake -E copy_if_different "$<TARGET_PDB_FILE:libntc>" "${params_OUTPUT_DIRECTORY}/")
#         
#         # Use generator expressions to select either the copy command or an empty string, depending on config
#         set(copy_command "$<$<OR:$<CONFIG:DEBUG>,$<CONFIG:RELWITHDEBINFO>>:${copy_command_base}>")
#         add_custom_command(TARGET ${params_TARGET} POST_BUILD COMMAND_EXPAND_LISTS COMMAND "${copy_command}")
#     endif()
# endfunction()
