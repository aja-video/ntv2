project(ntv2capture8k)

message("added ntv2capture8k")

set(NTV2CAPTURE8K_TARGET_INCLUDE_DIRS
	${AJA_LIBRARIES_ROOT}
	${AJA_APPS_ROOT}/crossplatform/demoapps
	${AJA_LIB_NTV2_ROOT}/includes
	${AJA_LIB_RETAIL_ROOT}/includes
	${AJA_LIB_RETAIL_ROOT}/services)

set(NTV2CAPTURE8K_HEADERS
	ntv2capture8k.h)
set(NTV2CAPTURE8K_SOURCES
	ntv2capture8k.cpp
	main.cpp)

set(DEMO_APPS_SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/../ntv2democommon.cpp)
set(DEMO_APPS_HEADERS
	${CMAKE_CURRENT_SOURCE_DIR}/../ntv2democommon.h)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(EXTRA_LIBRARIES dl pthread rt)
endif()

set(NTV2CAPTURE8K_TARGET_SOURCES
	${NTV2CAPTURE8K_SOURCES}
	${NTV2CAPTURE8K_HEADERS}
	${DEMO_APPS_SOURCES}
	${DEMO_APPS_HEADERS})

add_executable(ntv2capture8k ${NTV2CAPTURE8K_TARGET_SOURCES})
add_dependencies(ntv2capture8k ajantv2)
target_include_directories(ntv2capture8k PUBLIC ${NTV2CAPTURE8K_TARGET_INCLUDE_DIRS})
target_link_libraries(ntv2capture8k PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${DEMO_APPS_HEADERS} DESTINATION ${AJA_INSTALL_INCLUDEDIR}/ajaapps/crossplatform/demoapps)
	install(FILES ${DEMO_APPS_SOURCES} DESTINATION ${AJA_INSTALL_INCLUDEDIR}/ajaapps/crossplatform/demoapps)
	install(FILES ${NTV2CAPTURE8K_HEADERS} DESTINATION ${AJA_INSTALL_INCLUDEDIR}/ajaapps/crossplatform/demoapps/ntv2capture8k)
	install(FILES ${NTV2CAPTURE8K_SOURCES} DESTINATION ${AJA_INSTALL_INCLUDEDIR}/ajaapps/crossplatform/demoapps/ntv2capture8k)
endif()