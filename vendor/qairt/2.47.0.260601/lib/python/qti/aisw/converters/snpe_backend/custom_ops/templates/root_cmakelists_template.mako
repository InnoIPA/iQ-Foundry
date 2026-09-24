<%doc>
# ==============================================================================
#
#  Copyright (c) 2021, 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

"""This template is used to generate the root CMakeList.txt for udo package.

Args from python:
   package_name (str): Package name.
   subdirs (list of str): A list of directory name to add_subdirectory in CMake.
"""</%doc>
#================================================================================
# Auto Generated Code for ${package_info.name}
#================================================================================
cmake_minimum_required( VERSION 3.14 )
project( ${package_info.name} )

set( ROOT_INCLUDES "${'${CMAKE_CURRENT_SOURCE_DIR}'}/include" "${'${CMAKE_CURRENT_SOURCE_DIR}'}/include/utils" )
if( DEFINED ENV{SNPE_ROOT} )
   list( APPEND ROOT_INCLUDES "$ENV{SNPE_ROOT}/include/SNPE" )
elseif( DEFINED ENV{ZDL_ROOT} )
   file( GLOB ZDL_X86_64 "$ENV{ZDL_ROOT}/x86_64*" )
   list( APPEND ROOT_INCLUDES "${'${ZDL_X86_64}'}/include/zdl" )
else()
   message( FATAL_ERROR "SNPE_ROOT: Please set SNPE_ROOT or ZDL_ROOT to obtain Udo headers necessary to compile the package" )
endif()

MACRO(SUBDIRLIST result curdir)
FILE(GLOB children RELATIVE ${'${curdir}'} ${'${curdir}'}/*)
  SET(dirlist "")
  FOREACH(child ${'${children}'})
    IF(IS_DIRECTORY ${'${curdir}'}/${'${child}'})
      LIST(APPEND dirlist ${'${child}'})
    ENDIF()
  ENDFOREACH()
  SET(${'${result}'} ${'${dirlist}'})
ENDMACRO()


SUBDIRLIST(SUBDIRS ${'${CMAKE_CURRENT_SOURCE_DIR}'}/jni/src)
FOREACH(subdir ${'${SUBDIRS}'})
   if( EXISTS ${'${CMAKE_CURRENT_SOURCE_DIR}'}/jni/src/${'${subdir}'}/CMakeLists.txt )
      if(subdir MATCHES "CPU" OR subdir MATCHES "GPU" OR subdir MATCHES "reg")
         add_subdirectory(${'${CMAKE_CURRENT_SOURCE_DIR}'}/jni/src/${'${subdir}'})
      endif()
      if(subdir MATCHES "DSP" AND DEFINED ENV{HEXAGON_SDK_ROOT})
         string(REGEX MATCH "[0-9]+$" HEXAGON_VERSION ${'${subdir}'})
         if(HEXAGON_VERSION GREATER_EQUAL 68)
            add_subdirectory(${'${CMAKE_CURRENT_SOURCE_DIR}'}/jni/src/${'${subdir}'})
         endif()
      endif()
   endif()
ENDFOREACH()
