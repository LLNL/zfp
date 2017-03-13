add_executable(diffusion-raw diffusion.cpp)
target_compile_definitions(diffusion-raw PUBLIC WITHOUT_COMPRESSION)
target_link_libraries(diffusion-raw zfp)

add_executable(diffusion-zfp diffusion.cpp)
target_link_libraries(diffusion-zfp zfp)

add_executable(pgm pgm.c)
target_link_libraries(pgm zfp)

add_executable(simple simple.c)
target_link_libraries(simple zfp)

add_executable(speed speed.c)
target_link_libraries(speed zfp)
