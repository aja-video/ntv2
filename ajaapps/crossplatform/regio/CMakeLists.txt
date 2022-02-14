project(regio)

message("added regio")

set(REGIO_TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes)

set(REGIO_HEADERS
    i2c.h)
set(REGIO_SOURCES
    i2c.cpp
    regio.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(EXTRA_LIBRARIES dl pthread rt)
endif()

set(REGIO_TARGET_SOURCES
    ${REGIO_HEADERS}
	${REGIO_SOURCES})

add_executable(regio ${REGIO_TARGET_SOURCES})
add_dependencies(regio ajantv2)
target_include_directories(regio PUBLIC ${REGIO_TARGET_INCLUDE_DIRS})
target_link_libraries(regio PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${REGIO_HEADERS} DESTINATION ${AJA_INSTALL_INCLUDEDIR}/ajaapps/crossplatform/regio)
	install(FILES ${REGIO_SOURCES} DESTINATION ${AJA_INSTALL_INCLUDEDIR}/ajaapps/crossplatform/regio)
endif()
