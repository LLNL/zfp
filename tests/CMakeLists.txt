if(BUILD_TESTING OR BUILD_TESTING_FULL)
  # testzfp
  add_executable(testzfp testzfp.cpp)
  target_link_libraries(testzfp zfp)
  target_compile_definitions(testzfp PRIVATE ${zfp_compressed_array_defs})
  add_test(NAME testzfp COMMAND testzfp)
  
  # testviews
  add_executable(testviews testviews.cpp)
  if(ZFP_WITH_OPENMP)
    target_link_libraries(testviews zfp OpenMP::OpenMP_C)
  else()
    target_link_libraries(testviews zfp)
  endif()
  target_compile_definitions(testviews PRIVATE ${zfp_compressed_array_defs})
  add_test(NAME testviews COMMAND testviews)
endif()

if(BUILD_TESTING_FULL)
  set(CMAKE_CXX_STANDARD 11)
  
  # CMAKE_SH-NOTFOUND needed for mingw builds
  if(MINGW)
    list(APPEND CMOCKA_ARGS "-DCMAKE_SH=CMAKE_SH-NOTFOUND")
    list(APPEND GTEST_ARGS "-DCMAKE_SH=CMAKE_SH-NOTFOUND")
  endif()
  
  # clone cmocka 1.1.0 into /build
  list(APPEND CMOCKA_ARGS "-DWITH_STATIC_LIB=ON;-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER};-DUNIT_TESTING=OFF")
  
  include(ExternalProject)
  ExternalProject_Add(
    cmocka_cloned
    GIT_REPOSITORY    https://gitlab.com/cmocka/cmocka.git
    GIT_TAG           cmocka-1.1.5
    SOURCE_DIR        "${CMAKE_BINARY_DIR}/cmocka-src"
    BINARY_DIR        "${CMAKE_BINARY_DIR}/cmocka-build"
    CMAKE_ARGS        "${CMOCKA_ARGS}"
    INSTALL_COMMAND   ""
    STEP_TARGETS build
    EXCLUDE_FROM_ALL TRUE
  )
  ExternalProject_Get_Property(cmocka_cloned source_dir binary_dir)
  
  # name static library cmocka, wire up against cmocka_cloned
  add_library(cmocka STATIC IMPORTED GLOBAL)
  
  # choose proper library path & extension
  if(MSVC)
    set(IMPORTED_LOCATION_PATH "${binary_dir}/src/${CMAKE_BUILD_TYPE}/cmocka-static.lib")
  else()
    set(IMPORTED_LOCATION_PATH "${binary_dir}/src/libcmocka-static.a")
  endif()
  set_property(TARGET cmocka
    PROPERTY
    IMPORTED_LOCATION "${IMPORTED_LOCATION_PATH}"
  )
  
  add_dependencies(cmocka cmocka_cloned)
  include_directories(${source_dir}/include)
  
  # include home dir so #include statements are clear in test files
  include_directories(${ZFP_SOURCE_DIR} ${ZFP_SOURCE_DIR}/include)
  # access to constants/ and utils/
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  
  # suppress warnings for all targets
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-unused-function)
  endif()
  # -Wno-variadic-macros was not working for gcc...revisit
  if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    add_compile_options(-Wno-gnu-zero-variadic-macro-arguments)
  endif()
  # suppress googletest warning "conversion from 'float' to 'testing::internal::BiggestInt', possible loss of data"
  if(MSVC)
    add_compile_options(/wd4244)
  endif()
  
  
  add_subdirectory(utils)
  add_subdirectory(src)
  
  if(BUILD_CFP)
    add_subdirectory(cfp)
  endif()
  
  if(BUILD_ZFORP)
    add_subdirectory(fortran)
  endif()
  
  # needed to compile gtest on MSVC
  if(MSVC)
    list(APPEND GTEST_ARGS "/D:_SILENCE_TR1_DEPRECATION_NAMESPACE_WARNING=1")
  endif()
  
  # TODO: spend time getting googletest to compile on MinGW
  # checksums are generated through C tests, no need to compile C++ tests
  if((NOT MINGW) AND (NOT DEFINED ZFP_OMP_TESTS_ONLY) AND (NOT PRINT_CHECKSUMS))
    # clone googletest into build/
    configure_file(CMakeLists.txt.in ${ZFP_BINARY_DIR}/tests/googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" ${GTEST_ARGS} .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${ZFP_BINARY_DIR}/tests/googletest-download
    )
  
    if(result)
      message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    # build gtest
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
      RESULT_VARIABLE result
      WORKING_DIRECTORY ${ZFP_BINARY_DIR}/tests/googletest-download
    )
    if(result)
      message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()
  
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  
    add_subdirectory(${ZFP_BINARY_DIR}/tests/googletest-src
      ${ZFP_BINARY_DIR}/tests/googletest-build
    )
  
    if(CMAKE_VERSION VERSION_LESS 2.8.11)
      include_directories("${gtest_SOURCE_DIR}/include")
    endif()
  
    # needed to compile zfp tests with gtest on MSVC
    if(MSVC)
      target_compile_definitions(gtest_main INTERFACE GTEST_LINKED_AS_SHARED_LIBRARY=1)
    endif()
  
    add_subdirectory(array)
  endif()
  
  option(ZFP_BUILD_TESTING_SMALL "Enable small-sized array testing" ON)
  if(ZFP_BUILD_TESTING_SMALL)
    foreach(D IN ITEMS 1 2 3 4)
      foreach(P IN ITEMS 32 64)
        add_test(NAME small-arrays-${D}d-fp${P} COMMAND testzfp small ${D}d fp${P})
      endforeach()
    endforeach()
  endif()
  
  option(ZFP_BUILD_TESTING_LARGE "Enable large-sized array testing" OFF)
  if(ZFP_BUILD_TESTING_LARGE)
    foreach(D IN ITEMS 1 2 3 4)
      foreach(P IN ITEMS 32 64)
        add_test(NAME large-arrays-${D}d-fp${P} COMMAND testzfp large ${D}d fp${P})
      endforeach()
    endforeach()
  endif()
  
  if(BUILD_ZFPY)
    add_subdirectory(python)
  endif()
endif()
