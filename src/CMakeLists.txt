if(ZFP_WITH_CUDA)
  enable_language(CUDA)
  SET(CMAKE_CXX_FLAGS_PREVIOUS ${CMAKE_CXX_FLAGS})
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC" )

  add_subdirectory(cuda_zfp)
  cuda_include_directories(${PROJECT_SOURCE_DIR}/include)
  # cuda_wrap_srcs(zfp OBJ zfp_cuda_backend_obj cuda_zfp/cuZFP.cu)
  set(CUZFP_SOURCE cuda_zfp/cuZFP.cu)
  SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_PREVIOUS})
  add_definitions(-DZFP_WITH_CUDA)
  set (CUDA_SEPARATE_COMPILE_COMPILATION_OPTION 
       CUDA_SEPARABLE_COMPILATION ON)

endif()

if(ZFP_WITH_HIP)
  # if(NOT DEFINED HIP_PATH)
  #   if(NOT DEFINED ENV{HIP_PATH})
  #       set(HIP_PATH "/home/jieyang/opt/hip" CACHE PATH "Path to which HIP has been installed")
  #   else()
  #       set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
  #   endif()
  # endif()

  SET(CMAKE_CXX_FLAGS_PREVIOUS ${CMAKE_CXX_FLAGS})
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC" )
  # set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
  # find_package(HIP QUIET)
  # if(HIP_FOUND)
  #     message(STATUS "Found HIP: " ${HIP_VERSION})
  # else()
  #     message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
  # endif()
  add_definitions(-DZFP_WITH_HIP)
  set(HIPZFP_SOURCE hip_zfp/hipZFP.cpp)
endif()

set(zfp_source
  zfp.c
  bitstream.c
  #traitsf.h traitsd.h block1.h block2.h block3.h block4.h
  encode1f.c encode1d.c encode1i.c encode1l.c
  decode1f.c decode1d.c decode1i.c decode1l.c
  encode2f.c encode2d.c encode2i.c encode2l.c
  decode2f.c decode2d.c decode2i.c decode2l.c
  encode3f.c encode3d.c encode3i.c encode3l.c
  decode3f.c decode3d.c decode3i.c decode3l.c
  encode4f.c encode4d.c encode4i.c encode4l.c
  decode4f.c decode4d.c decode4i.c decode4l.c
  ${CUZFP_SOURCE} ${HIPZFP_SOURCE})

if(ZFP_WITH_HIP)
  set_source_files_properties(${zfp_source} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
endif()

# add_library(zfp ${zfp_source}
#                 ${zfp_cuda_backend_obj})

if(ZFP_WITH_HIP)
  hip_add_library(zfp SHARED ${zfp_source})
  add_library(zfp::zfp ALIAS zfp)
  # hip_add_library(zfp::zfp ALIAS zfp)
else()
  add_library(zfp ${zfp_source})
  add_library(zfp::zfp ALIAS zfp)
endif()


if(ZFP_WITH_CUDA)
  set_target_properties(zfp PROPERTIES
    ${CUDA_SEPARATE_COMPILE_COMPILATION_OPTION}
  )
endif()

if(ZFP_WITH_OPENMP)
  target_compile_options(zfp PRIVATE ${OpenMP_C_FLAGS})
  target_link_libraries(zfp PRIVATE ${OpenMP_C_LIBRARIES})
endif()

if(HAVE_LIBM_MATH)
  target_link_libraries(zfp PRIVATE m)
endif()

if(WIN32 AND BUILD_SHARED_LIBS)
  # Define ZFP_SOURCE when compiling libzfp to export symbols to Windows DLL
  list(APPEND zfp_public_defs ZFP_SHARED_LIBS)
  list(APPEND zfp_private_defs ZFP_SOURCE)
endif()

if(ZFP_WITH_CUDA)
  target_link_libraries(zfp PRIVATE ${CUDA_CUDART_LIBRARY} stdc++)
endif()

if(ZFP_WITH_HIP)
  find_package(CUDA)
  message("cjy7117:${CUDA_CUDART_LIBRARY}")
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

if(ZFP_WITH_HIP)
  set_property(TARGET zfp PROPERTY NO_SONAME 1)
endif()


install(TARGETS zfp EXPORT zfp-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
