add_library(cfp cfparrays.cpp)

if(DEFINED CFP_NAMESPACE)
  list(APPEND cfp_public_defs "CFP_NAMESPACE=${CFP_NAMESPACE}")
endif()

list(APPEND cfp_private_defs ${zfp_compressed_array_defs})

if(WIN32)
  # define ZFP_SOURCE when compiling libcfp to export symbols to Windows DLL
  list(APPEND cfp_private_defs ZFP_SOURCE)
endif()

target_compile_definitions(cfp
  PUBLIC ${cfp_public_defs}
  PRIVATE ${cfp_private_defs})

target_include_directories(cfp
  PUBLIC
    $<BUILD_INTERFACE:${ZFP_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${ZFP_SOURCE_DIR}/cfp/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${ZFP_SOURCE_DIR}/array
    ${ZFP_SOURCE_DIR}/src
)

target_link_libraries(cfp zfp)

set_property(TARGET cfp PROPERTY VERSION ${ZFP_VERSION})
set_property(TARGET cfp PROPERTY SOVERSION ${ZFP_VERSION_MAJOR})
set_property(TARGET cfp PROPERTY OUTPUT_NAME ${ZFP_LIBRARY_PREFIX}cfp)

install(TARGETS cfp EXPORT cfp-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
