project(ntv2thermo)

message("added ntv2thermo")

set(NTV2THERMO_TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes)

set(NTV2THERMO_SOURCES main.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(EXTRA_LIBRARIES dl pthread rt)
endif()

set(NTV2THERMO_TARGET_SOURCES
	${NTV2THERMO_SOURCES})

add_executable(ntv2thermo ${NTV2THERMO_TARGET_SOURCES})
add_dependencies(ntv2thermo ajantv2)
target_include_directories(ntv2thermo PUBLIC ${NTV2THERMO_TARGET_INCLUDE_DIRS})
target_link_libraries(ntv2thermo PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${NTV2THERMO_SOURCES} DESTINATION ${AJA_INSTALL_INCLUDEDIR}/ajaapps/crossplatform/ntv2thermo)
endif()
