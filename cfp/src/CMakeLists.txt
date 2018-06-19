add_library(cfp cfparrays.cpp)
target_include_directories(cfp
  PUBLIC
    ${ZFP_SOURCE_DIR}/include
    ${ZFP_SOURCE_DIR}/cfp/include
  PRIVATE
    ${ZFP_SOURCE_DIR}/array
    ${ZFP_SOURCE_DIR}/src
)

target_link_libraries(cfp zfp)
