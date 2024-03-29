project(ntv2firmwareinstaller)

set(DEMO_APPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../demoapps)

set(TARGET_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}/../
	${AJA_LIBRARIES_ROOT}
	${AJA_LIB_NTV2_ROOT}/includes
	${DEMO_APPS_ROOT}
	${AJA_LIB_NTV2_ROOT}/utilityfiles/private
	${AJA_LIB_NTV2_ROOT}/utilityfiles/public)

set(DEMO_APPS_HEADERS
	${DEMO_APPS_ROOT}/ntv2democommon.h)
set(DEMO_APPS_SOURCES
	${DEMO_APPS_ROOT}/ntv2democommon.cpp)
set(AJA_NTV2_UTILITY_PUBLIC_SOURCES
	${AJA_LIB_NTV2_ROOT}/utilityfiles/public/ntv2firmwareinstallerthread.h
	${AJA_LIB_NTV2_ROOT}/utilityfiles/public/ntv2firmwareinstallerthread.cpp)
set(NTV2FIRMWAREINSTALLER_SOURCES
	main.cpp)

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
	# noop
elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
	find_library(FOUNDATION_FRAMEWORK Foundation)
	set(TARGET_LINK_LIBS ${FOUNDATION_FRAMEWORK})
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
	set(TARGET_LINK_LIBS dl pthread rt)
endif()

set(TARGET_SOURCES
	${DEMO_APPS_HEADERS}
	${DEMO_APPS_SOURCES}
	${AJA_NTV2_UTILITY_PUBLIC_SOURCES}
	${NTV2FIRMWAREINSTALLER_SOURCES})

add_executable(ntv2firmwareinstaller ${TARGET_SOURCES})
add_dependencies(ntv2firmwareinstaller ajantv2)
target_include_directories(ntv2firmwareinstaller PUBLIC ${TARGET_INCLUDE_DIRS})
target_link_libraries(ntv2firmwareinstaller PUBLIC ajantv2 ${TARGET_LINK_LIBS})

if (AJA_INSTALL_SOURCES)
	install(FILES ${DEMO_APPS_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/demoapps)
	install(FILES ${DEMO_APPS_SOURCES} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/demoapps)
	install(FILES ${AJA_NTV2_UTILITY_PUBLIC_SOURCES} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajalibraries/ajantv2/utilityfiles/public)
	install(FILES ${NTV2FIRMWAREINSTALLER_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/ntv2firmwareinstaller)
	install(FILES ${NTV2FIRMWAREINSTALLER_SOURCES} DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/ntv2firmwareinstaller)
	install(FILES CMakeLists.txt DESTINATION ${CMAKE_INSTALL_PREFIX}/ajaapps/crossplatform/ntv2firmwareinstaller)
endif()

install(TARGETS ntv2firmwareinstaller
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
	FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR}
	PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if (AJA_CODE_SIGN)
    aja_code_sign(ntv2firmwareinstaller)
endif()
