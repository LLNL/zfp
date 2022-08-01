enable_language(Fortran)

if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
  set(dialect "-ffree-form -fimplicit-none")
  set(bounds "-fbounds-check")
endif()
if(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
  set(dialect "-stand -free -implicitnone")
  set(bounds "-check bounds")
endif()

set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR}/modules)
set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} ${bounds}")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${dialect}")

add_library(zFORp zfp.f90)
target_link_libraries(zFORp PRIVATE zfp)

set_property(TARGET zFORp PROPERTY VERSION ${ZFP_VERSION})
set_property(TARGET zFORp PROPERTY SOVERSION ${ZFP_VERSION_MAJOR})
set_property(TARGET zFORp PROPERTY OUTPUT_NAME ${ZFP_LIBRARY_PREFIX}zFORp)

# install location for module file
install(FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/zfp.mod
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install location for library
install(TARGETS zFORp EXPORT cFORp-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
