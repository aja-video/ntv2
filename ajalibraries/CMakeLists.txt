# ajaconv
if (EXISTS ${AJA_LIB_CONV_ROOT})
	if (NOT AJA_BUILD_OPENSOURCE)
		add_subdirectory(ajaconv)
		if (AJA_BUILD_TESTS)
			add_subdirectory(ajaconv/test)
		endif()
	endif()
endif()

# ajantv2
if (EXISTS ${AJA_LIB_NTV2_ROOT})
	add_subdirectory(ajantv2)
	if (AJA_BUILD_TESTS)
		add_subdirectory(ajabase/test)
		add_subdirectory(ajantv2/test)
	endif()
endif()

# ajacc
if (EXISTS ${AJA_LIB_CC_ROOT}
	AND EXISTS ${AJA_LIB_NTV2_ROOT})
	if (NOT AJA_BUILD_OPENSOURCE)
		add_subdirectory(ajacc)
		if (AJA_BUILD_TESTS)
			add_subdirectory(ajacc/test)
		endif()
	endif()
endif()

# ajagui
if (EXISTS ${AJA_LIB_GUI_ROOT}
	AND EXISTS ${AJA_LIB_NTV2_ROOT})
	if (NOT AJA_BUILD_OPENSOURCE)
		add_subdirectory(ajainternal/gui)
	endif()
endif()

# ajastreams
if (EXISTS ${AJA_LIB_STREAMS_ROOT}
	AND EXISTS ${AJA_LIB_NTV2_ROOT}
	AND EXISTS ${AJA_LIB_CONV_ROOT})

	if (NOT AJA_BUILD_OPENSOURCE)
		# Drastic deck control lib (AKA vvwext2)
		add_subdirectory(streams/drasticproj)

		add_subdirectory(streams)

		if (AJA_BUILD_TESTS)
			add_subdirectory(streams/test)
		endif()
	endif()
endif()
