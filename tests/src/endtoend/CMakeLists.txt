function(zfp_add_test dims type bits)
  if(NOT DEFINED ZFP_OMP_TESTS_ONLY)
    set(serial_test_name testZfpSerial${dims}d${type})
    add_executable(${serial_test_name} ${serial_test_name}.c)
    target_link_libraries(${serial_test_name}
      cmocka zfp zfpHashLib genSmoothRandNumsLib stridedOperationsLib
      zfpChecksumsLib zfpCompressionParamsLib zfpTimerLib)
    if(HAVE_LIBM_MATH)
      target_link_libraries(${serial_test_name} m)
    endif()

    target_compile_definitions(${serial_test_name} PRIVATE ${zfp_private_defs})
    if(PRINT_CHECKSUMS)
      target_compile_definitions(${serial_test_name} PUBLIC PRINT_CHECKSUMS)
    endif()

    add_test(NAME ${serial_test_name} COMMAND ${serial_test_name})
  endif()

  if(ZFP_WITH_OPENMP)
    set(omp_test_name testZfpOmp${dims}d${type})
    add_executable(${omp_test_name} ${omp_test_name}.c)
    target_compile_definitions(${omp_test_name} PRIVATE ${zfp_private_defs})
    target_link_libraries(${omp_test_name}
      cmocka zfp zfpHashLib genSmoothRandNumsLib stridedOperationsLib
      zfpChecksumsLib zfpTimerLib zfpCompressionParamsLib
      OpenMP::OpenMP_C)
    if(HAVE_LIBM_MATH)
      target_link_libraries(${omp_test_name} m)
    endif()
    add_test(NAME ${omp_test_name} COMMAND ${omp_test_name})
    set_property(TEST ${omp_test_name} PROPERTY RUN_SERIAL TRUE)
  endif()

  if(NOT DEFINED ZFP_OMP_TESTS_ONLY)
    if(ZFP_WITH_CUDA)
      add_definitions(-DZFP_WITH_CUDA)

      set(cuda_test_name testZfpCuda${dims}d${type})
      add_executable(${cuda_test_name} ${cuda_test_name}.c)
      target_compile_definitions(${cuda_test_name} PRIVATE ${zfp_private_defs})
      target_link_libraries(${cuda_test_name}
        cmocka zfp zfpHashLib genSmoothRandNumsLib stridedOperationsLib
        zfpChecksumsLib zfpTimerLib zfpCompressionParamsLib)
      if(HAVE_LIBM_MATH)
        target_link_libraries(${cuda_test_name} m)
      endif()
      add_test(NAME ${cuda_test_name} COMMAND ${cuda_test_name})
      set_property(TEST ${cuda_test_name} PROPERTY RUN_SERIAL TRUE)
    endif()
  endif()
endfunction()

zfp_add_test(1 Int32 32)
zfp_add_test(1 Int64 64)
zfp_add_test(2 Int32 32)
zfp_add_test(2 Int64 64)
zfp_add_test(3 Int32 32)
zfp_add_test(3 Int64 64)
zfp_add_test(4 Int32 32)
zfp_add_test(4 Int64 64)

zfp_add_test(1 Float 32)
zfp_add_test(1 Double 64)
zfp_add_test(2 Float 32)
zfp_add_test(2 Double 64)
zfp_add_test(3 Float 32)
zfp_add_test(3 Double 64)
zfp_add_test(4 Float 32)
zfp_add_test(4 Double 64)
