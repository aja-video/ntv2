# AJA NTV2 SDK (MIT license)

## Building the ajantv2 library with [CMake](https://cmake.org/) and [ninja build](https://ninja-build.org/) on Windows/macOS/Linux

1. Download and install CMake 3.10 or higher and place it in your *PATH* on the filesystem. CMake is available on all three major platforms supported by ajantv2 (Windows, macOS, Linux).
    + **Windows** - Download and install the CMake x64 .msi installer:
	https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-windows-x86_64.msi

    + **macOS** - Install the [Homebrew](https://brew.sh/) package manager:
    ```
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    ```
    Then, run: `brew install cmake`
    
	+ **Ubuntu Linux 20.04** - Install CMake via the apt package manager:
    ```
	sudo apt update
    sudo apt install cmake
    ```

1. Download ninja build, unzip and place it somewhere within the *PATH* on your filesystem.
    + The latest release (v1.10.2) for Windows, macOS and Linux can be downloaded from GitHub: https://github.com/ninja-build/ninja/releases/tag/v1.10.2

1. Clone the ntv2 git repository.

    ```
    git clone git@github.com:aja-video/ntv2.git
    ```

1. `cd` into the `ntv2` git repo directory.

1. Create a temporary build directory where CMake will produce the build artifacts. For example: `mkdir cmake-build`

1. `cd` into the new build directory: `cd cmake-build`

1. Windows/MSVC-only: Run the `vcvarsall.bat` script within the VS2017 or 2019 installation to initialize the MSVC environment for x86 64-bit.
    ```
    "c:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
    ```

1. Run `cmake` to generate the input files for build system to consume.

    CMake is a "meta-build" tool and as such it does not build the _ajantv2_ library itself. Rather, it _generates_ input files that another build system consumes to build _ajantv2_. This tutorial uses the aformentioned "ninja build" system but other CMake build system "generators" are available on Windows, macOS and Linux. See https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html for more information about CMake generators.

	+ Generating build files for the ajantv2 library:
    ```
    cmake -DCMAKE_BUILD_TYPE=Debug -GNinja ..    
    ```
    **NOTE:** Specify `-DCMAKE_BUILD_TYPE=Release` to build the ajantv2 library in release mode with optimizations.

1. Run ninja to build the ntv2 software:
    ```
    ninja -f build.ninja
    ```

Binaries of the ajantv2 library can be found within your temporary build directory, under `ajalibraries/ajantv2`.

### Part 2: Building with CMake + Visual Studio Solution files

These instructions assume that you have:
+ Installed CMake 3.10 or higher.
+ Installed either Visual Studio 2017 or 2019 and any Visual C++ dependencies required to build ntv2.
+ Checked out the ntv2 git repository.

1. Create temporary build directory:
    ```
    mkdir cmake-build
    ```
1. `cd` into the temporary build directory.
1. Run CMake to generate a Visual Studio Solution:
    + VS 2017 (x64)
	```
	cmake -DCMAKE_BUILD_TYPE=Debug -G "Visual Studio 15 2017 Win64" ..
	```
	+ VS 2019 (x64)
	```
	cmake -DCMAKE_BUILD_TYPE=Debug -G "Visual Studio 16 2019" -A Win64 ..
	```
1. Build the generated VS solution with MSBuild:
    ```
    msbuild /p:Platform=x64 /p:Configuration=Debug /t:Clean,Build ntv2.sln
    ```
    *NOTE*: The solution (ntv2.sln) file may also be opened in Visual Studio for building.

### Part 3: Building with CMake + Unix Makefiles on macOS or Linux

These instructions assume that you have:
+ Installed CMake 3.10 or higher.
+ Installed GCC or Clang on your macOS/Linux system.
+ Checked out the ntv2 git repository.

1. Create temporary build directory:
    ```
    mkdir cmake-build
    ```
1. `cd` into the temporary build directory.
1. Run CMake to generate Makefiles for ntv2:
    ```
	cmake -DCMAKE_BUILD_TYPE=Debug -G Makefile ..
    ```
1. Build the generated Makefiles:
    ```
	make -j$(nproc)
	```


## Deploying NTV2 sources and build artifacts with CMake Install

The `cmake --install` command can be used to deploy NTV2 sources and build artifacts to default system install  
directories or to variable paths, specified by the following variables:
- `AJA_INSTALL_INCLUDEDIR` - Destination for deploying NTV2 sources/headers
- `AJA_INSTALL_LIBDIR` - Destination for deploying built static/shared libs
- `AJA_INSTALL_BINDIR` - Destination for deploying built executables
- `AJA_INSTALL_FRAMEWORKDIR` - Destination for deploying built macOS Frameworks

If these variables are not overridden at CMake build time, the default CMake install paths will be used.

See `cmake/CommonVars.cmake` for more info.

### Linux example
```
#!/bin/bash
NTV2_DIR=$PWD
NTV2_INSTALL_DIR=ntv2-install
QT_DIR=/opt/Qt5.13.2/5.13.2/gcc_64

rm -rf ${NTV2_INSTALL_DIR} && \
rm -rf ninja && mkdir ninja && pushd ninja && \
cmake -DCMAKE_BUILD_TYPE=Debug -GNinja \
-DAJA_INSTALL_HEADERS=ON -DAJA_INSTALL_SOURCES=ON \
-DAJA_INSTALL_LIBDIR="${NTV2_DIR}/${NTV2_INSTALL_DIR}/lib" \
-DAJA_INSTALL_BINDIR="${NTV2_DIR}/${NTV2_INSTALL_DIR}/bin" \
-DAJA_INSTALL_FRAMEWORKDIR="${NTV2_DIR}/${NTV2_INSTALL_DIR}/lib" \
-DAJA_INSTALL_INCLUDEDIR="${NTV2_DIR}/${NTV2_INSTALL_DIR}" \
-DAJA_DEPLOY_LIBS=ON -DAJA_QT_DIR=${QT_DIR} \
-DAJA_BUILD_OPENSOURCE=ON .. && \
ninja -f build.ninja && \
cmake --install ajaapps && \
cmake --install ajadriver && \
cmake --install ajalibraries/ajantv2 && \
popd
```