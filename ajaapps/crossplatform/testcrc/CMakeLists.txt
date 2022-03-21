project(testcrc)

message("added testcrc")

set(TESTCRC_TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes)

set(TESTCRC_SOURCES
    testcrc.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(EXTRA_LIBRARIES dl pthread rt)
endif()

set(TESTCRC_TARGET_SOURCES
	${TESTCRC_SOURCES})

add_executable(testcrc ${TESTCRC_TARGET_SOURCES})
add_dependencies(testcrc ajantv2)
target_include_directories(testcrc PUBLIC ${TESTCRC_TARGET_INCLUDE_DIRS})
target_link_libraries(testcrc PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${TESTCRC_SOURCES} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/testcrc)
endif()

install(TARGETS testcrc
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
