add_executable(zfpcmd zfp.c)
set_property(TARGET zfpcmd PROPERTY OUTPUT_NAME zfp)
target_link_libraries(zfpcmd zfp)
