project(pciwhacker)

message("added pciwhacker")

set(DEMO_APPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../demoapps)

set(PCIWHACKER_TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes
    ${DEMO_APPS_ROOT})

set(DEMO_APPS_SOURCES
	${DEMO_APPS_ROOT}/ntv2democommon.cpp)
set(DEMO_APPS_HEADERS
	${DEMO_APPS_ROOT}/ntv2democommon.h)

set(PCIWHACKER_SOURCES
    main.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(EXTRA_LIBRARIES ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(EXTRA_LIBRARIES dl pthread rt)
endif()

set(PCIWHACKER_TARGET_SOURCES
	${DEMO_APPS_HEADERS}
	${DEMO_APPS_SOURCES}
	${PCIWHACKER_SOURCES})

add_executable(pciwhacker ${PCIWHACKER_TARGET_SOURCES})
add_dependencies(pciwhacker ajantv2)
target_include_directories(pciwhacker PUBLIC ${PCIWHACKER_TARGET_INCLUDE_DIRS})
target_link_libraries(pciwhacker PUBLIC ajantv2 ${EXTRA_LIBRARIES})

if (AJA_INSTALL_SOURCES)
	install(FILES ${DEMO_APPS_HEADERS} DESTINATION ${AJA_INSTALL_DIR}/ajaapps/crossplatform/demoapps)
	install(FILES ${DEMO_APPS_SOURCES} DESTINATION ${AJA_INSTALL_DIR}/ajaapps/crossplatform/demoapps)
	install(FILES ${PCIWHACKER_SOURCES} DESTINATION ${AJA_INSTALL_DIR}/ajaapps/crossplatform/pciwhacker)
endif()

install(TARGETS pciwhacker
	ARCHIVE DESTINATION ${AJA_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${AJA_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${AJA_INSTALL_BINDIR}
	FRAMEWORK DESTINATION ${AJA_INSTALL_FRAMEWORKDIR})
