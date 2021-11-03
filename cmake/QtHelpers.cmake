# https://stackoverflow.com/questions/60854495/qt5-cmake-include-all-libraries-into-executable
# get absolute path to qmake, then use it to find windeployqt executable
function(deploy_qt_libs target)
    if (WIN32)
        find_package(Qt5Core REQUIRED)
        get_target_property(_qmake_executable Qt5::qmake IMPORTED_LOCATION)
        get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)

        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${_qt_bin_dir}/windeployqt.exe"
                --verbose 1
                --no-svg
                --no-angle
                --no-opengl
                --no-opengl-sw
                --no-compiler-runtime
                --no-system-d3d-compiler
                --release
                \"$<TARGET_FILE:${target}>\"
            COMMENT "Deploying Qt Release libraries for target '${target}' via winqtdeploy ..."
        )

        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND "${_qt_bin_dir}/windeployqt.exe"
                    --verbose 1
                    --no-svg
                    --no-angle
                    --no-opengl
                    --no-opengl-sw
                    --no-compiler-runtime
                    --no-system-d3d-compiler
                    --debug
                    \"$<TARGET_FILE:${target}>\"
                COMMENT "Deploying Qt Debug libraries for target '${target}' via winqtdeploy ..."
            )
        endif()
    elseif (APPLE)
        find_package(Qt5Core REQUIRED)
        get_target_property(_qmake_executable Qt5::qmake IMPORTED_LOCATION)
        get_filename_component(_qt_bin_dir "${_qmake_executable}" DIRECTORY)

        get_target_property(_is_target_bundle ${target} MACOSX_BUNDLE)
        if (_is_target_bundle)
            message("macOS bundle: ${target}")
            add_custom_target(
                ${target}_macqtdeploy ALL
                DEPENDS $<TARGET_BUNDLE_DIR:${target}>
                COMMAND ${_qt_bin_dir}/macdeployqt $<TARGET_BUNDLE_DIR:${target}>/ -always-overwrite
                COMMENT "Deploying Qt Frameworks into .app bundle for target '${target}' via macqtdeploy ..."
                VERBATIM
            )
        else()
            message("macOS binary: ${target}")
            add_custom_target(
                ${target}_macqtdeploy ALL
                DEPENDS $<TARGET_FILE_DIR:${target}>
                COMMAND ${_qt_bin_dir}/macdeployqt $<TARGET_FILE_DIR:${target}>/ -executable=$<TARGET_FILE_DIR:${target}>/${target} -always-overwrite
                COMMENT "Deploying Qt Frameworks into for target '${target}' via macqtdeploy ..."
                VERBATIM
            )
        endif()

        add_dependencies(${target}_macqtdeploy ${target})
    elseif (LINUX)
        message("lin_deploy_qt TODO")
        return()
    endif()
endfunction()
