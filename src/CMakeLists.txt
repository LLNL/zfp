if(ZFP_WITH_CUDA)
  SET(CMAKE_CXX_FLAGS_PREVIOUS ${CMAKE_CXX_FLAGS})
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC" )

  add_subdirectory(cuda_zfp)
  cuda_include_directories(${PROJECT_SOURCE_DIR}/include)
  cuda_wrap_srcs(zfp OBJ zfp_cuda_backend_obj cuda_zfp/cuZFP.cu)
  SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_PREVIOUS})
  add_definitions(-DZFP_WITH_CUDA)
endif()


set(zfp_source
  zfp.c
  bitstream.c
  traitsf.h traitsd.h block1.h block2.h block3.h block4.h
  encode1f.c encode1d.c encode1i.c encode1l.c
  decode1f.c decode1d.c decode1i.c decode1l.c
  encode2f.c encode2d.c encode2i.c encode2l.c
  decode2f.c decode2d.c decode2i.c decode2l.c
  encode3f.c encode3d.c encode3i.c encode3l.c
  decode3f.c decode3d.c decode3i.c decode3l.c
  encode4f.c encode4d.c encode4i.c encode4l.c
  decode4f.c decode4d.c decode4i.c decode4l.c)

add_library(zfp ${zfp_source}
                ${zfp_cuda_backend_obj})
add_library(zfp::zfp ALIAS zfp)

if(ZFP_WITH_OPENMP)
  target_compile_options(zfp PRIVATE ${OpenMP_C_FLAGS})
  target_link_libraries(zfp PRIVATE ${OpenMP_C_LIBRARIES})
endif()

if(HAVE_LIBM_MATH)
  target_link_libraries(zfp PRIVATE m)
endif()

if(WIN32)
  # Define ZFP_SOURCE when compiling libzfp to export symbols to Windows DLL
  list(APPEND zfp_private_defs ZFP_SOURCE)
endif()

if(ZFP_WITH_CUDA)
  target_link_libraries(zfp PRIVATE ${CUDA_CUDART_LIBRARY} stdc++)
endif()

target_compile_definitions(zfp
  PRIVATE ${zfp_private_defs}
  PUBLIC ${zfp_public_defs}
)

target_include_directories(zfp
  PUBLIC
    $<BUILD_INTERFACE:${ZFP_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  INTERFACE
    $<BUILD_INTERFACE:${ZFP_SOURCE_DIR}/array>)

set_property(TARGET zfp PROPERTY VERSION ${ZFP_VERSION})
set_property(TARGET zfp PROPERTY SOVERSION ${ZFP_VERSION_MAJOR})
set_property(TARGET zfp PROPERTY OUTPUT_NAME ${ZFP_LIBRARY_PREFIX}zfp)

install(TARGETS zfp EXPORT zfp-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
