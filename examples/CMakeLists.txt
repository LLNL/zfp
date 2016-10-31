# Prefer the static lib for the examples
if(BUILD_STATIC)
    set(_zfp_lib zfp_static)
else()
    set(_zfp_lib zfp_shared)
endif()

# diffusion-raw
add_executable(diffusion-raw diffusion.cpp)
target_link_libraries(diffusion-raw ${_zfp_lib})
target_include_directories(diffusion-raw PUBLIC ${CMAKE_SOURCE_DIR}/array)
target_compile_definitions(diffusion-raw PUBLIC -DWITHOUT_COMPRESSION)

# diffusion-zfp
add_executable(diffusion-zfp diffusion.cpp)
target_link_libraries(diffusion-zfp ${_zfp_lib})
target_include_directories(diffusion-zfp PUBLIC ${CMAKE_SOURCE_DIR}/array)

# pgm
add_executable(pgm pgm.c)
target_link_libraries(pgm ${_zfp_lib} m)

# simple
add_executable(simple simple.c)
target_link_libraries(simple ${_zfp_lib} m)

# speed
add_executable(speed speed.c)
target_link_libraries(speed ${_zfp_lib} m)

# testzfp
add_executable(testzfp testzfp.cpp fields.c)
add_test(NAME testzfp COMMAND testzfp)
target_link_libraries(testzfp ${_zfp_lib})
target_include_directories(testzfp PUBLIC ${CMAKE_SOURCE_DIR}/array)

# zfp
add_executable(zfp-bin zfp.c)
set_target_properties(zfp-bin PROPERTIES OUTPUT_NAME zfp)
target_link_libraries(zfp-bin ${_zfp_lib} m)

