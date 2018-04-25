# - Config file for the zfp package
#
# It defines the following variables
#  ZFP_INCLUDE_DIRS - include directories for zfp
#  ZFP_LIBRARIES    - libraries to link against
#
# And the following imported targets:
#   zfp::zfp
#

include("${CMAKE_CURRENT_LIST_DIR}/zfp-config-version.cmake")

include(FindPackageHandleStandardArgs)
set(${CMAKE_FIND_PACKAGE_NAME}_CONFIG "${CMAKE_CURRENT_LIST_FILE}")
find_package_handle_standard_args(${CMAKE_FIND_PACKAGE_NAME} CONFIG_MODE)

if(NOT TARGET zfp::zfp)
  include("${CMAKE_CURRENT_LIST_DIR}/zfp-targets.cmake")
endif()

set(ZFP_LIBRARIES zfp::zfp)
set(ZFP_INCLUDE_DIRS
  $<TARGET_PROPERTY:zfp::zfp,INTERFACE_INCLUDE_DIRECTORIES>
)
