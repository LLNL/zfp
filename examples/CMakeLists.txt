add_executable(diffusion diffusion.cpp)
target_link_libraries(diffusion zfp)
target_compile_definitions(diffusion PRIVATE ${zfp_defs})

add_executable(inplace inplace.c)
target_link_libraries(inplace zfp)
target_compile_definitions(inplace PRIVATE ${zfp_defs})

add_executable(iterator iterator.cpp)
target_link_libraries(iterator zfp)
target_compile_definitions(iterator PRIVATE ${zfp_defs})

add_executable(pgm pgm.c)
target_link_libraries(pgm zfp)
target_compile_definitions(pgm PRIVATE ${zfp_defs})

add_executable(simple simple.c)
target_link_libraries(simple zfp)
target_compile_definitions(simple PRIVATE ${zfp_defs})

add_executable(speed speed.c)
target_link_libraries(speed zfp)
target_compile_definitions(speed PRIVATE ${zfp_defs})
