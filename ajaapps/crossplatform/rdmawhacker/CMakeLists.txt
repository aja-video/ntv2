project(rdmawhacker)

message("added rdmawhacker")

set(DEMO_APPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../demoapps)

set(RDMAWHACKER_TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes
    ${DEMO_APPS_ROOT})

set(DEMO_APPS_SOURCES
	${DEMO_APPS_ROOT}/ntv2democommon.cpp)
set(DEMO_APPS_HEADERS
	${DEMO_APPS_ROOT}/ntv2democommon.h)

set(RDMAWHACKER_SOURCES
    rdmawhacker.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(RDMAWHACKER_TARGET_DEFINES
        -DAJA_RDMA)
    set(EXTRA_LIBRARIES cuda cudart dl pthread rt)
endif()

set(RDMAWHACKER_TARGET_SOURCES
	${RDMAWHACKER_SOURCES})

# add_executable(rdmawhacker ${RDMAWHACKER_TARGET_SOURCES})
# add_dependencies(rdmawhacker ajantv2)
# target_compile_definitions(rdmawhacker ${RDMAWHACKER_TARGET_DEFINES})
# target_include_directories(rdmawhacker PUBLIC ${RDMAWHACKER_TARGET_INCLUDE_DIRS})
# target_link_libraries(rdmawhacker PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${RDMAWHACKER_SOURCES} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/rdmawhacker)
endif()

# install(TARGETS rdmawhacker
# 	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
# 	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
# 	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
# 	FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
