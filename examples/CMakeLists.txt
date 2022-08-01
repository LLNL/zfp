add_executable(array array.cpp)
target_compile_definitions(array PRIVATE ${zfp_compressed_array_defs})
target_link_libraries(array zfp)

add_executable(diffusion diffusion.cpp)
target_compile_definitions(diffusion PRIVATE ${zfp_compressed_array_defs})
if(ZFP_WITH_OPENMP)
  target_link_libraries(diffusion zfp OpenMP::OpenMP_CXX)
else()
  target_link_libraries(diffusion zfp)
endif()

if(BUILD_CFP)
  add_executable(diffusionC diffusionC.c)
  target_link_libraries(diffusionC cfp)
endif()

add_executable(inplace inplace.c)
target_link_libraries(inplace zfp)

add_executable(iterator iterator.cpp)
target_link_libraries(iterator zfp)
target_compile_definitions(iterator PRIVATE ${zfp_compressed_array_defs})

if(BUILD_CFP)
  add_executable(iteratorC iteratorC.c)
  target_link_libraries(iteratorC cfp)
endif()

add_executable(pgm pgm.c)
target_link_libraries(pgm zfp)

add_executable(ppm ppm.c)
target_link_libraries(ppm zfp)
target_compile_definitions(ppm PRIVATE ${ppm_private_defs})

add_executable(simple simple.c)
target_link_libraries(simple zfp)

add_executable(speed speed.c)
target_link_libraries(speed zfp)

if(HAVE_LIBM_MATH)
  target_link_libraries(array m)
  target_link_libraries(diffusion m)

  if(BUILD_CFP)
    target_link_libraries(diffusionC m)
  endif()

  target_link_libraries(inplace m)
  target_link_libraries(pgm m)
  target_link_libraries(ppm m)
  target_link_libraries(simple m)
endif()
