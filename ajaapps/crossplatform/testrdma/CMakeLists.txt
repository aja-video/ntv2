find_package(CUDA 11.2)
if (NOT CUDA_FOUND)
	aja_message(STATUS "skipping testrdma (CUDA not found)")
	return()
endif()

project(testrdma LANGUAGES CUDA CXX)

set(TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes
	${CUDA_INCLUDE_DIRS})

set(TESTRDMA_SOURCES
    testrdma.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(TARGET_LINK_LIBS ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(TARGET_COMPILE_DEFS
		-DAJA_RDMA)
	set(TARGET_LINK_LIBS cuda ${CUDA_LIBRARIES} dl pthread rt)
endif()

set(TARGET_SOURCES
	${TESTRDMA_SOURCES})

add_executable(testrdma ${TARGET_SOURCES})
add_dependencies(testrdma ajantv2)
target_compile_definitions(testrdma PUBLIC ${TARGET_COMPILE_DEFS})
target_include_directories(testrdma PUBLIC ${TARGET_INCLUDE_DIRS})
target_link_libraries(testrdma PUBLIC ajantv2 ${TARGET_LINK_LIBS})

if (AJA_INSTALL_SOURCES)
	install(FILES ${TESTRDMA_SOURCES} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/testrdma)
	install(FILES CMakeLists.txt DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/testrdma)
endif()

install(TARGETS testrdma
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
	FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(AJA_CODE_SIGN)
	aja_code_sign(testrdma)
endif()
