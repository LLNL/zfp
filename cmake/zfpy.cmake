function(zfpy_add_module name)
    # Cythonize
    add_cython_target(${name} ${zfpy_SOURCE_DIR}/${name}.pyx CXX PY3)
    add_library(${name} MODULE ${${name}})
    target_include_directories(${name} PRIVATE ${ZFP_SOURCE_DIR}/include ${NumPy_INCLUDE_DIR})
    target_link_libraries(${name} zfp)
    set_property(TARGET ${name} APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:zfp>")
    python_extension_module(${name})
    set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})

    # Install to the typical python module directory
    set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
    install(TARGETS ${name} LIBRARY DESTINATION ${python_install_lib_dir})
endfunction()
