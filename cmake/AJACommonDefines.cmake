# Common preprocessor defines
if(CMAKE_BUILD_TYPE MATCHES Debug)
    aja_message(STATUS "Build Type: Debug")
    list(APPEND AJANTV2_COMPILE_DEFINITIONS -DAJA_DEBUG -D_DEBUG)
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    aja_message(STATUS "Build Type: Release with Debug Symbols")
else()
    aja_message(STATUS "Build Type: Release")
    list(APPEND AJANTV2_COMPILE_DEFINITIONS -DNDEBUG)
endif()
# Platform-specific preprocessor defines
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    list(APPEND AJANTV2_COMPILE_DEFINITIONS
        -DAJA_WINDOWS
        -DMSWindows
        -D_WINDOWS
        -D_CONSOLE
        -DUNICODE
        -D_UNICODE
        -DWIN32_LEAN_AND_MEAN
        -D_CRT_SECURE_NO_WARNINGS
        -D_SCL_SECURE_NO_WARNINGS)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND AJANTV2_COMPILE_DEFINITIONS
        -DAJALinux
        -DAJA_LINUX)
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    list(APPEND AJANTV2_COMPILE_DEFINITIONS
        -DAJAMac
        -DAJA_MAC
        -D__STDC_CONSTANT_MACROS)
endif()

add_definitions(${AJANTV2_COMPILE_DEFINITIONS})
