# ajaconv
if (EXISTS ${AJA_LIB_CONV_ROOT})
	if (NOT AJA_BUILD_OPENSOURCE)
		aja_add_subdirectory(ajaconv)
		if (AJA_BUILD_TESTS)
			aja_add_subdirectory(ajaconv/test)
		endif()
	endif()
endif()

# ajantv2
if (EXISTS ${AJA_LIB_NTV2_ROOT})
    aja_add_subdirectory(ajantv2)
	if (AJA_BUILD_TESTS)
		aja_add_subdirectory(ajabase/test)
		aja_add_subdirectory(ajantv2/test)
	endif()
endif()

# ajacc
if (EXISTS ${AJA_LIB_CC_ROOT}
	AND EXISTS ${AJA_LIB_NTV2_ROOT})
	if (NOT AJA_BUILD_OPENSOURCE)
		aja_add_subdirectory(ajacc)
		if (AJA_BUILD_TESTS)
			aja_add_subdirectory(ajacc/test)
		endif()
	endif()
endif()

# ajagui
if (EXISTS ${AJA_LIB_GUI_ROOT}
	AND EXISTS ${AJA_LIB_NTV2_ROOT})
	if (NOT AJA_BUILD_OPENSOURCE)
		aja_add_subdirectory(ajainternal/gui)
	endif()
endif()

# ajastreams
if (EXISTS ${AJA_LIB_STREAMS_ROOT}
	AND EXISTS ${AJA_LIB_NTV2_ROOT}
	AND EXISTS ${AJA_LIB_CONV_ROOT})
	if (NOT AJA_BUILD_OPENSOURCE)
		# Drastic deck control lib (AKA vvwexts2)
		aja_add_subdirectory(streams/drasticproj "Drastic Technologies Ltd.")
		aja_add_subdirectory(streams)
		if (AJA_BUILD_TESTS)
			aja_add_subdirectory(streams/test)
		endif()
	endif()
endif()

if (EXISTS ${AJA_LIB_DOCS_ROOT}
	AND EXISTS ${AJA_LIB_NTV2_ROOT}
	AND AJA_BUILD_DOCS)
	aja_add_subdirectory(docs)
endif()
