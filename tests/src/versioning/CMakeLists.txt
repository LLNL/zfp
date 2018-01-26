if(ZFP_WITH_VERSION_PREFIX)
  # compile only
  add_executable(testZfpPrefix testZfpPrefix.c)
  target_link_libraries(testZfpPrefix zfp)

  if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /WX")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
  endif()
endif()

if(ZFP_MULTIPLE_VERSIONS_PARENT AND ZFP_V4_BINARY_DIR)
  add_executable(testZfpApi testZfpApi.c)
  add_test(NAME testZfpApi COMMAND testZfpApi)
  target_link_libraries(testZfpApi cmocka zfp hash32Lib genSmoothRandNumsLib)

  if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /WX")
  else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
  endif()
endif()
