add_library(cfp cfparrays.cpp)
if(DEFINED CFP_NAMESPACE)
  target_compile_definitions(cfp PUBLIC "CFP_NAMESPACE=${CFP_NAMESPACE}")
endif()
target_include_directories(cfp
  PUBLIC
    ${ZFP_SOURCE_DIR}/include
    ${ZFP_SOURCE_DIR}/cfp/include
  PRIVATE
    ${ZFP_SOURCE_DIR}/array
    ${ZFP_SOURCE_DIR}/src
)

target_link_libraries(cfp zfp)
