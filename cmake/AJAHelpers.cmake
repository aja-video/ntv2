function(aja_message mode msg)
	message(${mode} "AJA:  ${msg}" ${ARGN})
endfunction(aja_message)

# wrapper around add_subdirectory that only adds the directory
# if the target exists and contains a CMakeLists.txt file
# NOTE: Has an optional variadic arg for specifying a full project name (eg. AJA ControlRoom, etc.)
function(aja_add_subdirectory target)
	set(AJA_PRODUCT_NAME "")
	set (variadic_args ${ARGN})
	list(LENGTH variadic_args variadic_count)
	if (${variadic_count} GREATER 0)
		list(GET variadic_args 0 AJA_PRODUCT_NAME)
	endif()
	if (${variadic_count} GREATER 1)
		list(GET variadic_args 1 AJA_PRODUCT_DESC)
	endif()
	set(target_path ${CMAKE_CURRENT_SOURCE_DIR}/${target})
	if (EXISTS ${target_path} AND EXISTS ${target_path}/CMakeLists.txt)
		add_subdirectory(${target_path})
		set(status_msg "added target: ${target}")
		if (AJA_PRODUCT_NAME)
			set(status_msg "${status_msg} (${AJA_PRODUCT_NAME})")
		endif()
		aja_message(STATUS ${status_msg})
	else()
		set(status_msg "target not found: ${target}")
		if (AJA_PRODUCT_NAME)
			set(status_msg "${status_msg} (${AJA_PRODUCT_NAME})")
		endif()
		aja_message(WARNING ${status_msg})
	endif()
endfunction(aja_add_subdirectory)

function(post_build_copy_file target src dst)
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${src} ${dst})
endfunction(post_build_copy_file)

function(aja_make_version_rc product_name product_desc file_in file_out icon_file)
    set(AJA_APP_ICON ${icon_file})
    set(AJA_PRODUCT_NAME ${product_name})
	set(AJA_PRODUCT_DESC ${product_desc})
    configure_file(${file_in} ${file_out} @ONLY)
endfunction(aja_make_version_rc)

function(aja_make_version_elf target product_name product_desc file_in file_out icon_file)
    set(AJA_APP_ICON ${icon_file})
    set(AJA_PRODUCT_NAME ${product_name})
	set(AJA_PRODUCT_DESC ${product_desc})
    configure_file(${file_in} ${file_out} @ONLY)
	if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
		# embed version info as JSON in elf header
		add_custom_command(TARGET ${target} POST_BUILD
			COMMAND
			objcopy
			--add-section "aja_app_info=${file_out}"
			"$<TARGET_FILE:${target}>"
			VERBATIM)
	endif()
endfunction(aja_make_version_elf)

function(aja_make_version_plist target product_name bundle_sig file_in file_out icon_file)
	set(AJA_PRODUCT_NAME ${product_name})
	set(AJA_BUNDLE_SIGNATURE ${bundle_sig})
	get_filename_component(icon_filename ${icon_file} NAME)
	set(AJA_APP_ICON ${icon_filename})
	aja_message(STATUS "mac icon: ${icon_filename}")
	configure_file(${file_in} ${file_out} @ONLY)
	set_source_files_properties(${file_out} PROPERTIES
		MACOSX_PACKAGE_LOCATION Contents)
	set_source_files_properties(${icon_file} PROPERTIES
		MACOSX_PACKAGE_LOCATION Resources)
    set_target_properties(${target} PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_INFO_PLIST ${file_out})
endfunction(aja_make_version_plist)

function(aja_code_sign targets)
	# Code sign build targets with aja "pythonlib" helper scripts.
	# NOTE: This functionality is not yet available in ntv2 open-source.
	if (UNIX AND NOT APPLE)
		aja_message(STATUS "Code signing is not available on this platform!")
		return()
	endif()

	if (WIN32)
		execute_process(COMMAND where python OUTPUT_VARIABLE python3_exe)
	elseif(APPLE)
		execute_process(COMMAND which python3 OUTPUT_VARIABLE python3_exe)
	endif()
	file(TO_CMAKE_PATH "${python3_exe}" python3_exe)
	string(REPLACE "\n" "" python3_exe ${python3_exe})
	if (NOT EXISTS ${python3_exe})
		aja_message(STATUS "Python 3 (${python3_exe}) not found! Code signing scripts not available.")
		return()
	endif()
	foreach(target IN LISTS targets)
		if (EXISTS "${AJA_NTV2_ROOT}/installers/pythonlib/aja/tools/sign.py")
			set(pythonlib_path ${AJA_NTV2_ROOT}/installers/pythonlib)
			get_filename_component(pythonlib_path "${pythonlib_path}" REALPATH)
			set(sign_script_path ${AJA_NTV2_ROOT}/installers/pythonlib/aja/tools/sign.py)
			get_filename_component(sign_script_path "${sign_script_path}" REALPATH)
			aja_message(STATUS "Code Sign: $<TARGET_FILE:${target}>")
			add_custom_command(TARGET ${target} POST_BUILD
				COMMAND
					${CMAKE_COMMAND} -E env "PYTHONPATH=\"${pythonlib_path}\""
					${python3_exe}
					${sign_script_path}
					$<TARGET_FILE:${target}>
				COMMENT "Signing '$<TARGET_FILE:${target}>' ...")
		endif()
	endforeach()
endfunction(aja_code_sign)
