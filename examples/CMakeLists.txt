add_executable(diffusion diffusion.cpp)
target_link_libraries(diffusion zfp)

add_executable(inplace inplace.c)
target_link_libraries(inplace zfp)

add_executable(iterator iterator.c)
target_link_libraries(iterator zfp)

add_executable(pgm pgm.c)
target_link_libraries(pgm zfp)

add_executable(simple simple.c)
target_link_libraries(simple zfp)

add_executable(speed speed.c)
target_link_libraries(speed zfp)
