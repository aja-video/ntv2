project(logreader)

message("added logreader")

set(LOGREADER_TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes)

set(LOGREADER_SOURCES main.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(EXTRA_LIBRARIES dl pthread rt)
endif()

set(LOGREADER_TARGET_SOURCES
	${LOGREADER_SOURCES})

add_executable(logreader ${LOGREADER_TARGET_SOURCES})
add_dependencies(logreader ajantv2)
target_include_directories(logreader PUBLIC ${LOGREADER_TARGET_INCLUDE_DIRS})
target_link_libraries(logreader PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${LOGREADER_SOURCES} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/logreader)
endif()

install(TARGETS logreader
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
