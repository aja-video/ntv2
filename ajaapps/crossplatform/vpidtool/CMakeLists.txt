project(vpidtool)

message("added vpidtool")

set(VPIDTOOL_TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes)

set(VPIDTOOL_SOURCES main.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(EXTRA_LIBRARIES dl pthread rt)
endif()

set(VPIDTOOL_TARGET_SOURCES
	${VPIDTOOL_SOURCES})

add_executable(vpidtool ${VPIDTOOL_TARGET_SOURCES})
add_dependencies(vpidtool ajantv2)
target_include_directories(vpidtool PUBLIC ${VPIDTOOL_TARGET_INCLUDE_DIRS})
target_link_libraries(vpidtool PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${VPIDTOOL_SOURCES} DESTINATION ${AJA_INSTALL_INCLUDEDIR}/ajaapps/crossplatform/vpidtool)
endif()
