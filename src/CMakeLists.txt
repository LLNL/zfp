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

  if (ZFP_V4_DIR)
    # construct paths to compiled zfpV4 object files
    SET(ZFP_V4_SRC_FILES
      zfp.c
      bitstream.c
      encode1f.c encode1d.c encode1i.c encode1l.c
      decode1f.c decode1d.c decode1i.c decode1l.c
      encode2f.c encode2d.c encode2i.c encode2l.c
      decode2f.c decode2d.c decode2i.c decode2l.c
      encode3f.c encode3d.c encode3i.c encode3l.c
      decode3f.c decode3d.c decode3i.c decode3l.c)

    if(MSVC)
      SET(ZFP_V4_OBJ_PREFIX "${ZFP_V4_BINARY_DIR}/src/zfpV4.dir/${CMAKE_BUILD_TYPE}")
    else()
      SET(ZFP_V4_OBJ_PREFIX "${ZFP_V4_BINARY_DIR}/src/CMakeFiles/zfpV4.dir")
    endif()

    SET(ZFP_V4_OBJS "")
    foreach(ZFP_V4_SRC_FILE ${ZFP_V4_SRC_FILES})
      # MSVC object files are of form: zfp.obj (remove .c)
      if(MSVC)
        string(REPLACE "\.c" "" ZFP_V4_SRC_FILE ${ZFP_V4_SRC_FILE})
      endif()

      LIST(APPEND ZFP_V4_OBJS "${ZFP_V4_OBJ_PREFIX}/${ZFP_V4_SRC_FILE}${CMAKE_C_OUTPUT_EXTENSION}")
    endforeach()

    set_source_files_properties(
      ${ZFP_V4_OBJS}
      PROPERTIES
      EXTERNAL_OBJECT TRUE
      GENERATED TRUE)
  endif()
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

  # ensure children are built first, so their object files exist
  if(ZFP_V4_DIR)
    add_dependencies(zfp zfp_v4-proj)
  endif()

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
