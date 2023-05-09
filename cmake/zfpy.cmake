function(zfpy_add_module name type)
    add_cython_target(${name} ${zfpy_SOURCE_DIR}/${name}.pyx ${type} PY3)
    add_library(${name} MODULE ${${name}})
    target_include_directories(${name} PRIVATE 
        ${NumPy_INCLUDE_DIR}
        ${ZFP_SOURCE_DIR}/include 
    )
    target_link_libraries(${name} zfp)
    set_property(TARGET ${name} APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:zfp>")
    set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})
    python_extension_module(${name})

    set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
    install(TARGETS ${name} LIBRARY DESTINATION ${python_install_lib_dir})
endfunction()

function(zfpy_add_test_module name src_dir)
    include_directories(${ZFP_SOURCE_DIR}/include)
    include_directories(${ZFP_SOURCE_DIR}/python)
    include_directories(${NumPy_INCLUDE_DIR})
    include_directories(${ZFP_SOURCE_DIR}/tests/python)
    include_directories(${ZFP_SOURCE_DIR}/tests/utils)
    include_directories(${ZFP_SOURCE_DIR})

    add_cython_target(${name} ${src_dir}/${name}.pyx C PY3)
    add_library(${name} MODULE ${${name}})
    target_link_libraries(${name} zfpChecksumsLib zfp genSmoothRandNumsLib stridedOperationsLib zfpCompressionParamsLib zfpHashLib)
    set_property(TARGET ${name} APPEND PROPERTY BUILD_RPATH "$<TARGET_FILE_DIR:zfp>")
    set_target_properties(${name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYLIB_BUILD_DIR})
    python_extension_module(${name})
    
    set(python_install_lib_dir "lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/")
    install(TARGETS ${name} LIBRARY DESTINATION ${python_install_lib_dir})   
endfunction()

function(zfpy_add_test name)
    if(MSVC)
      set(TEST_PYTHON_PATH "${TEST_PYTHON_PATH}/${CMAKE_BUILD_TYPE}")
    endif()
    
    if(DEFINED ENV{PYTHONPATH})
      set(TEST_PYTHON_PATH "${TEST_PYTHON_PATH}:$ENV{PYTHONPATH}")
    endif()

    add_test(NAME ${name}
      COMMAND ${PYTHON_EXECUTABLE} ${name}.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    
    set_tests_properties(${name} PROPERTIES
      ENVIRONMENT PYTHONPATH=${TEST_PYTHON_PATH})
endfunction()
