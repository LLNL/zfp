# clone cmocka 1.1.0 into /build
include(ExternalProject)
ExternalProject_Add(
  cmocka_cloned
  GIT_REPOSITORY    git://git.cryptomilk.org/projects/cmocka.git
  GIT_TAG           cmocka-1.1.0
  SOURCE_DIR        "${CMAKE_BINARY_DIR}/cmocka-src"
  BINARY_DIR        "${CMAKE_BINARY_DIR}/cmocka-build"
  CMAKE_ARGS        -DWITH_STATIC_LIB=ON -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_SH=CMAKE_SH-NOTFOUND -DUNIT_TESTING=OFF
  BUILD_COMMAND     ${CMAKE_COMMAND} --build . WORKING_DIRECTORY ${CMAKE_BINARY_DIR} cmocka_static
  INSTALL_COMMAND   ""
)
ExternalProject_Get_Property(cmocka_cloned source_dir binary_dir)

# name static library cmocka, wire up against cmocka_cloned
add_library(cmocka STATIC IMPORTED GLOBAL)

# choose proper library path & extension
if(MSVC)
  set(IMPORTED_LOCATION_PATH "${binary_dir}/src/Debug/cmocka.lib")
else()
  set(IMPORTED_LOCATION_PATH "${binary_dir}/src/libcmocka.a")
endif()
set_property(TARGET cmocka
  PROPERTY
  IMPORTED_LOCATION "${IMPORTED_LOCATION_PATH}"
)

add_dependencies(cmocka cmocka_cloned)
include_directories(${source_dir}/include)

# include home dir so #include statements are clear in test files
include_directories(${ZFP_SOURCE_DIR} ${ZFP_SOURCE_DIR}/include)

add_subdirectory(src)


add_executable(testzfp testzfp.cpp fields.c)
target_link_libraries(testzfp zfp)

option(ZFP_BUILD_TESTING_SMALL "Enable small-sized array testing" ON)
if(ZFP_BUILD_TESTING_SMALL)
  foreach(D IN ITEMS 1 2 3)
    foreach(P IN ITEMS 32 64)
      add_test(NAME small-arrays-${D}d-fp${P} COMMAND testzfp small ${D}d fp${P})
    endforeach()
  endforeach()
endif()

option(ZFP_BUILD_TESTING_MEDIUM "Enable medium-sized array testing" OFF)
if(ZFP_BUILD_TESTING_MEDIUM)
  foreach(D IN ITEMS 1 2 3)
    foreach(P IN ITEMS 32 64)
      add_test(NAME medium-arrays-${D}d-fp${P} COMMAND testzfp medium ${D}d fp${P})
    endforeach()
  endforeach()
endif()

option(ZFP_BUILD_TESTING_LARGE "Enable large array testing" OFF)
if(ZFP_BUILD_TESTING_LARGE)
  foreach(D IN ITEMS 1 2 3)
    foreach(P IN ITEMS 32 64)
      add_test(NAME large-arrays-${D}d-fp${P} COMMAND testzfp large ${D}d fp${P})
    endforeach()
  endforeach()
endif()
