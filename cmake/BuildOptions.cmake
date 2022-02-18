set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_DEBUG_POSTFIX d)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# macOS build settings
if (NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.12" CACHE STRING "Minimum macOS deployment version" FORCE)
endif()
# TODO(paulh): generate universal builds up front
# set(CMAKE_OSX_ARCHITECTURES arm64 x86_64)
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	# get the Mac OS SDK major and minor numbers
	# MACOS_SDK_NAME will contain something like MacOSX11.1
	# MACOS_SDK_VERSION = 11.1
	# MACOS_SDK_VERSION_MAJOR = 11
	# MACOS_SDK_VERSION_MINOR = 1
	get_filename_component(MACOS_SDK_NAME ${CMAKE_OSX_SYSROOT} NAME_WLE)
	string(REPLACE "MacOSX" "" MACOS_SDK_VERSION ${MACOS_SDK_NAME})
	string(REPLACE "." ";" MACOS_SDK_VERSION_LIST ${MACOS_SDK_VERSION})
	list(GET MACOS_SDK_VERSION_LIST 0 MACOS_SDK_VERSION_MAJOR)
	list(GET MACOS_SDK_VERSION_LIST 1 MACOS_SDK_VERSION_MINOR)

    set(MACOS_FRAMEWORKS_DIR ${CMAKE_OSX_SYSROOT}/System/Library/Frameworks)
endif()

if (AJA_QT_DIR)
    message("AJA Qt Dir Variable: ${AJA_QT_DIR}")
endif()
if (DEFINED ENV{AJA_QT_DIR})
    set(AJA_QT_DIR $ENV{AJA_QT_DIR})
    message("AJA Qt Dir Environment: $ENV{AJA_QT_DIR}")
endif()

option(AJA_BUILDING_CMAKE "Build NTV2 SDK with version.h generated by CMake" OFF)
option(AJA_BUILD_OPENSOURCE "Build NTV2 SDK as open-source (exclude internal apps/libs)" ON)
option(AJA_BUILD_QT_BASED "Build NTV2 Demos and Apps which depend upon Qt" ON)
option(AJA_BUILD_SHARED "Build NTV2 shared libraries" OFF)
option(AJA_DEPLOY_OUTPUTS "Gather build artifacts into top-level directory, and in bin/lib dirs." OFF)
option(AJA_INSTALL_SOURCES "Deploy sources into build output directory." OFF)
option(AJA_INSTALL_HEADERS "Deploy headers into build output directory." OFF)
option(AJA_DEPLOY_LIBS "Deploy dependency libraries (DLL/dylib) into the .exe output directory (Win32) or .app bundle (macOS)." OFF)

option(AJA_BUILD_APPS "Build AJA NTV2 applications" ON)
option(AJA_BUILD_DRIVER "Build AJA NTV2 driver" ON)
option(AJA_BUILD_LIBS "Build AJA libraries" ON)
option(AJA_BUILD_PLUGINS "Build AJA Plug-ins" ON)
option(AJA_BUILD_TESTS "Build unit tests" ON)
option(AJA_BUILD_QA "Build AJA NTV2 QA apps and libs" ON)
option(AJA_BUILD_NONAJA "Build non-aja shared libraries" ON)

# Build ajantv2 open-source SDK
if (AJA_BUILD_OPENSOURCE)
    message("NOTE: Building open-source AJA NTV2 SDK components (MIT license)")
    set(AJA_BUILD_APPS ON)
    set(AJA_BUILD_DRIVER ON)
    set(AJA_BUILD_LIBS ON)
    set(AJA_BUILD_PLUGINS OFF)
    set(AJA_BUILD_TESTS OFF)
    set(AJA_BUILD_QA OFF)
	set(AJA_BUILD_NONAJA OFF)
endif()
