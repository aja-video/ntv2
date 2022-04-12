project(rdmawhacker)


set(DEMO_APPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../demoapps)

set(TARGET_INCLUDE_DIRS
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
	set(TARGET_LINK_LIBS ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(TARGET_COMPILE_DEFS
        -DAJA_RDMA)
    set(TARGET_LINK_LIBS cuda cudart dl pthread rt)
endif()

set(TARGET_SOURCES
	${RDMAWHACKER_SOURCES})

# add_executable(rdmawhacker ${TARGET_SOURCES})
# add_dependencies(rdmawhacker ajantv2)
# target_compile_definitions(rdmawhacker ${TARGET_COMPILE_DEFS})
# target_include_directories(rdmawhacker PUBLIC ${TARGET_INCLUDE_DIRS})
# target_link_libraries(rdmawhacker PUBLIC ajantv2 ${TARGET_LINK_LIBS})

if (AJA_INSTALL_SOURCES)
	install(FILES ${RDMAWHACKER_SOURCES} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/rdmawhacker)
endif()

# install(TARGETS rdmawhacker
# 	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
# 	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
# 	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# 	FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX})
