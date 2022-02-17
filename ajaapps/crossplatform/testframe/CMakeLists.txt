project(testframe)

message("added testframe")

set(TESTFRAME_TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes)

set(TESTFRAME_SOURCES
	testframe.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(EXTRA_LIBRARIES dl pthread rt)
endif()

set(TESTFRAME_TARGET_SOURCES
	${TESTFRAME_SOURCES})

add_executable(testframe ${TESTFRAME_TARGET_SOURCES})
add_dependencies(testframe ajantv2)
target_include_directories(testframe PUBLIC ${TESTFRAME_TARGET_INCLUDE_DIRS})
target_link_libraries(testframe PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${TESTFRAME_SOURCES} DESTINATION ${AJA_INSTALL_INCLUDEDIR}/ajaapps/crossplatform/testframe)
endif()
