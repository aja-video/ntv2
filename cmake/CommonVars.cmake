include(GNUInstallDirs)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(AJA_BITS 64)
    message(STATUS "Bits: 64-bit")
else()
    set(AJA_BITS 32)
    message(STATUS "Bits: 32-bit")
endif()

# cmake --install configuration
# Use default "CMAKE_INSTALL_<DIR>" variables unless overridden at command-line
if (DEFINED AJA_INSTALL_DIR)
    set(AJA_INSTALL_DIR CACHE STRING ${CMAKE_INSTALL_INCLUDEDIR})
else()
    set(AJA_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
endif()
message(STATUS "AJA_INSTALL_DIR = ${AJA_INSTALL_DIR}")

if (DEFINED AJA_INSTALL_LIBDIR)
    set(AJA_INSTALL_LIBDIR CACHE STRING ${CMAKE_INSTALL_LIBDIR})
else()
    set(AJA_INSTALL_LIBDIR ${CMAKE_INSTALL_LIBDIR})
endif()
message(STATUS "AJA_INSTALL_LIBDIR = ${AJA_INSTALL_LIBDIR}")

if (DEFINED AJA_INSTALL_BINDIR)
    set(AJA_INSTALL_BINDIR CACHE STRING ${CMAKE_INSTALL_BINDIR})
else()
    set(AJA_INSTALL_BINDIR ${CMAKE_INSTALL_BINDIR})
endif()
message(STATUS "AJA_INSTALL_BINDIR = ${AJA_INSTALL_BINDIR}")

if (DEFINED AJA_INSTALL_FRAMEWORKDIR)
    set(AJA_INSTALL_FRAMEWORKDIR CACHE STRING ${CMAKE_INSTALL_PREFIX})
else()
    set(AJA_INSTALL_FRAMEWORKDIR ${CMAKE_INSTALL_PREFIX})
endif()
message(STATUS "AJA_INSTALL_FRAMEWORKDIR = ${AJA_INSTALL_FRAMEWORKDIR}")
message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")
