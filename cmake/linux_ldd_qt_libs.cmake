function(linux_ldd_qt_libs _exe_path _svg_support _qt_ver)
    # Run ldd to determine which Qt libs the executable is linked against.
    set(_qt_libs
        "libQt5XcbQpa.so.5"
        "libQt5DBus.so.5"
        "libicui18n.so.56"
        "libicuuc.so.56"
        "libicudata.so.56")
    if (_svg_support)
        list(APPEND _qt_libs "libQt5Svg.so.5")
    endif()
    execute_process(COMMAND /usr/bin/ldd ${_exe_path} OUTPUT_VARIABLE _ldd_stdout)

    # split lines in ldd stdout result
    string(REPLACE "\n" ";" _ldd_lines "${_ldd_stdout}")

    foreach(_ldd_line IN LISTS _ldd_lines)
        if (_ldd_line)
            # split line by "=>" substring and find "libQt${_qt_ver}" string piece
            string(REPLACE "=>" ";" _ldd_parts "${_ldd_line}")
            string(FIND "${_ldd_parts}" "libQt${_qt_ver}" _qt_ver_found)
        else()
            set(_qt_ver_found -1)
        endif()
        # split Qt lib name and append to list of Qt libs
        if (${_qt_ver_found} EQUAL 1)
            list(GET _ldd_parts 0 _qt_lib_name)
            string(REPLACE "\t" "" _qt_lib_name "${_qt_lib_name}")
            string(REPLACE " " "" _qt_lib_name "${_qt_lib_name}")
            list(APPEND _qt_libs ${_qt_lib_name})
        endif()
    endforeach()

    foreach(_ldd IN LISTS _qt_libs)
        message(${_ldd})
    endforeach()
endfunction()

linux_ldd_qt_libs(${TARGET_NAME} ${SVG_SUPPORT} ${QT_MAJOR_VERSION})
