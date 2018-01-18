set(zfp_source
  zfp.c
  bitstream.c
  traitsf.h traitsd.h block1.h block2.h block3.h
  encode1f.c encode1d.c encode1i.c encode1l.c
  decode1f.c decode1d.c decode1i.c decode1l.c
  encode2f.c encode2d.c encode2i.c encode2l.c
  decode2f.c decode2d.c decode2i.c decode2l.c
  encode3f.c encode3d.c encode3i.c encode3l.c
  decode3f.c decode3d.c decode3i.c decode3l.c)

if(ZFP_MULTIPLE_VERSIONS_PARENT)
  list(APPEND zfp_source versioning/zfpApi.c)

  SET(ZFP_V4_OBJ_PREFIX "${ZFP_V4_BINARY_DIR}/src/CMakeFiles/zfpV4.dir")
  SET(ZFP_V4_OBJS
    ${ZFP_V4_OBJ_PREFIX}/zfp.c.o
    ${ZFP_V4_OBJ_PREFIX}/bitstream.c.o

    ${ZFP_V4_OBJ_PREFIX}/encode1f.c.o
    ${ZFP_V4_OBJ_PREFIX}/encode1d.c.o
    ${ZFP_V4_OBJ_PREFIX}/encode1i.c.o
    ${ZFP_V4_OBJ_PREFIX}/encode1l.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode1f.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode1d.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode1i.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode1l.c.o

    ${ZFP_V4_OBJ_PREFIX}/encode2f.c.o
    ${ZFP_V4_OBJ_PREFIX}/encode2d.c.o
    ${ZFP_V4_OBJ_PREFIX}/encode2i.c.o
    ${ZFP_V4_OBJ_PREFIX}/encode2l.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode2f.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode2d.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode2i.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode2l.c.o

    ${ZFP_V4_OBJ_PREFIX}/encode3f.c.o
    ${ZFP_V4_OBJ_PREFIX}/encode3d.c.o
    ${ZFP_V4_OBJ_PREFIX}/encode3i.c.o
    ${ZFP_V4_OBJ_PREFIX}/encode3l.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode3f.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode3d.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode3i.c.o
    ${ZFP_V4_OBJ_PREFIX}/decode3l.c.o)

  set_source_files_properties(
    ${ZFP_V4_OBJS}
    PROPERTIES
    EXTERNAL_OBJECT TRUE
    GENERATED TRUE)
endif()

# externally viewed, final library name is always zfp
if(ZFP_MULTIPLE_VERSIONS_CHILD)
  add_library(zfpV5 OBJECT ${zfp_source})

  target_compile_definitions(zfpV5
    PUBLIC ${zfp_public_defs}
    PRIVATE ${zfp_defs})

  target_include_directories(zfpV5
    PUBLIC    ${ZFP_SOURCE_DIR}/include
    INTERFACE ${ZFP_SOURCE_DIR}/array)

  set_property(TARGET zfpV5 PROPERTY VERSION ${ZFP_VERSION})
  set_property(TARGET zfpV5 PROPERTY SOVERSION 0)

else()
  add_library(zfp STATIC ${zfp_source} ${ZFP_V4_OBJS})

  if(UNIX)
    target_link_libraries(zfp PUBLIC m)
  endif()

  install(TARGETS zfp
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

  target_compile_definitions(zfp
    PUBLIC ${zfp_public_defs}
    PRIVATE ${zfp_defs})

  target_include_directories(zfp
    PUBLIC    ${ZFP_SOURCE_DIR}/include
    INTERFACE ${ZFP_SOURCE_DIR}/array)

  set_property(TARGET zfp PROPERTY VERSION ${ZFP_VERSION})
  set_property(TARGET zfp PROPERTY SOVERSION 0)
endif()

if(WIN32)
  # Define ZFP_SOURCE when compiling libzfp to export symbols to Windows DLL
  list(APPEND zfp_defs ZFP_SOURCE)
endif()
