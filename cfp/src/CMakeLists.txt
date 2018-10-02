add_library(cfp cfparrays.cpp)

if(DEFINED CFP_NAMESPACE)
  list(APPEND cfp_public_defs "CFP_NAMESPACE=${CFP_NAMESPACE}")
endif()

if(WIN32)
  # define ZFP_SOURCE when compiling libcfp to export symbols to Windows DLL
  list(APPEND cfp_private_defs ZFP_SOURCE)
endif()

target_compile_definitions(cfp
  PUBLIC ${cfp_public_defs}
  PRIVATE ${cfp_private_defs})

target_include_directories(cfp
  PUBLIC
    ${ZFP_SOURCE_DIR}/include
    ${ZFP_SOURCE_DIR}/cfp/include
  PRIVATE
    ${ZFP_SOURCE_DIR}/array
    ${ZFP_SOURCE_DIR}/src
)

target_link_libraries(cfp zfp)
