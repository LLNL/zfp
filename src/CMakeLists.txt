set(zfp_defs)

if(NOT (ZFP_BIT_STREAM_WORD_SIZE EQUAL 64))
  list(APPEND zfp_defs BIT_STREAM_WORD_TYPE=${ZFP_BIT_STREAM_WORD_SIZE})
endif()

if(ZFP_WITH_BIT_STREAM_STRIDED)
  list(APPEND zfp_defs BIT_STREAM_STRIDED)
endif()

if(NOT ZFP_WITH_COMPRESSION)
  list(APPEND zfp_defs WITHOUT_COMPRESSION)
endif()

if(ZFP_WITH_ALIGNED_ALLOC)
  list(APPEND zfp_defs ALIGNED_ALLOC)
endif()

if(ZFP_WITH_CACHE_TWOWAY)
  list(APPEND zfp_defs CACHE_TWOWAY)
endif()

if(ZFP_WITH_CACHE_FASH_HASH)
  list(APPEND zfp_defs CACHE_FAST_HASH)
endif()

if(ZFP_WITH_CACHE_PROFILING)
  list(APPEND zfp_defs CACHE_PROFILING)
endif()

set(zfp_source
  zfp.c
  bitstream.c
  traitsf.h traitsd.h

  block1.h encode1f.c encode1d.c decode1f.c decode1d.c
  block2.h encode2f.c encode2d.c decode2f.c decode2d.c
  block3.h encode3f.c encode3d.c decode3f.c decode3d.c)

add_library(zfp ${zfp_source})
if(UNIX)
  target_link_libraries(zfp PUBLIC m)
endif()

target_compile_definitions(zfp PRIVATE ${zfp_defs})

target_include_directories(zfp
  PUBLIC    ${ZFP_SOURCE_DIR}/inc
  INTERFACE ${ZFP_SOURCE_DIR}/array)

set_property(TARGET zfp PROPERTY VERSION ${ZFP_VERSION})
set_property(TARGET zfp PROPERTY SOVERSION 0)

install(TARGETS zfp
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
