project(testrdma)

message("added testrdma")

set(TESTRDMA_TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes)

set(TESTRDMA_SOURCES
    testrdma.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(EXTRA_LIBRARIES cuda cudart dl pthread rt)
endif()

set(TESTRDMA_TARGET_SOURCES
	${TESTRDMA_SOURCES})

# add_executable(testrdma ${TESTRDMA_TARGET_SOURCES})
# add_dependencies(testrdma ajantv2)
# target_include_directories(testrdma PUBLIC ${TESTRDMA_TARGET_INCLUDE_DIRS})
# target_link_libraries(testrdma PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${TESTRDMA_SOURCES} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/testrdma)
endif()

# install(TARGETS testrdma
# 	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
# 	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
# 	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# 	FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX})
