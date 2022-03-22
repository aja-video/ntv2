# wrapper around add_subdirectory that only adds the directory
# if the target exists and contains a CMakeLists.txt file
function(add_subdirectory_if_valid target)
	if (EXISTS ${target} AND EXISTS ${target}/CMakeLists.txt)
		add_subdirectory(${target})
	endif()
endfunction()

function(post_build_copy_file target src dst)
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${src} ${dst})
endfunction()
