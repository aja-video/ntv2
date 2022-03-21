project(supportlog)

message("added supportlog")

set(SUPPORTLOG_TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes)

set(SUPPORTLOG_SOURCES
    main.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(EXTRA_LIBRARIES dl pthread rt)
endif()

set(SUPPORTLOG_TARGET_SOURCES
	${SUPPORTLOG_SOURCES})

add_executable(supportlog ${SUPPORTLOG_TARGET_SOURCES})
add_dependencies(supportlog ajantv2)
target_include_directories(supportlog PUBLIC ${SUPPORTLOG_TARGET_INCLUDE_DIRS})
target_link_libraries(supportlog PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${SUPPORTLOG_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/supportlog)
	install(FILES ${SUPPORTLOG_SOURCES} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/supportlog)
endif()

install(TARGETS supportlog
	ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
	RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
	FRAMEWORK DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
