# wrapper around add_subdirectory that only adds the directory
# if the target exists and contains a CMakeLists.txt file
function(add_subdirectory_if_valid target)
	if (EXISTS ${target} AND EXISTS ${target}/CMakeLists.txt)
		add_subdirectory(${target})
	endif()
endfunction()