set(AJA_COMPANY_NAME "AJA Video Systems, Inc.")
set(AJA_WEBSITE "https://www.aja.com/")

# NTV2 SDK version number variables. Generates an include file in `ajantv2/includes/ntv2version.h`,
# which is used throughout the SDK via `ajantv2/includes/ntv2enums.h`.
# Override the following variables to set an arbitrary NTV2 SDK version number.
string(TIMESTAMP AJA_BUILD_MONTH "%m")
string(TIMESTAMP AJA_BUILD_DAY "%d")
string(TIMESTAMP AJA_BUILD_YEAR "%Y")
string(TIMESTAMP DATETIME_NOW "%m/%d/%Y +8:%H:%M:%S")
if (NOT AJA_NTV2_SDK_VERSION_MAJOR)
    set(AJA_NTV2_SDK_VERSION_MAJOR "16")
endif()
if (NOT AJA_NTV2_SDK_VERSION_MINOR)
    set(AJA_NTV2_SDK_VERSION_MINOR "3")
endif()
if (NOT AJA_NTV2_SDK_VERSION_POINT)
    set(AJA_NTV2_SDK_VERSION_POINT "1")
endif()

set(BUILD_NUMBER_CACHE
    ${CMAKE_MODULE_PATH}.CMakeBuildNumber
    CACHE INTERNAL "AJA NTV2 build number cache file")
if (NOT AJA_NTV2_SDK_BUILD_NUMBER AND EXISTS ${BUILD_NUMBER_CACHE})
    file(READ ${BUILD_NUMBER_CACHE} AJA_NTV2_SDK_BUILD_NUMBER)
    math(EXPR AJA_NTV2_SDK_BUILD_NUMBER "${AJA_NTV2_SDK_BUILD_NUMBER}+1")
elseif(NOT DEFINED AJA_NTV2_SDK_BUILD_NUMBER)
    set(AJA_NTV2_SDK_BUILD_NUMBER "1")
endif()
set(AJA_NTV2_SDK_BUILD_DATETIME ${DATETIME_NOW})
set(AJA_NTV2_VER_STR "${AJA_NTV2_SDK_VERSION_MAJOR}.${AJA_NTV2_SDK_VERSION_MINOR}.${AJA_NTV2_SDK_VERSION_POINT}")
set(AJA_NTV2_VER_STR_LONG "${AJA_NTV2_VER_STR}.${AJA_NTV2_SDK_BUILD_NUMBER}")
string(REPLACE "." "," AJA_NTV2_VER_STR_COMMA "${AJA_NTV2_VER_STR_LONG}")
set(AJA_NTV2_SDK_BUILD_TYPE "d") # r = release, d = debug, b = beta
aja_message(STATUS "NTV2 SDK Version: ${AJA_NTV2_VER_STR} ${aja_ntv2_build_type_letter}${AJA_NTV2_SDK_BUILD_NUMBER}")
