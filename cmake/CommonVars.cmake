include(GNUInstallDirs)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(AJA_BITS 64)
    message(STATUS "Bits: 64-bit")
else()
    set(AJA_BITS 32)
    message(STATUS "Bits: 32-bit")
endif()
