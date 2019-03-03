add_executable(zfpcmd zfp.c)

# protect against LNK1114: cannot overwrite the original file 'lib/Release/zfp.lib'; error code 32;
# rationale: linker can't handle the case of an executable file having the same name as a library file;
# this is not an issue when BUILD_SHARED_LIBS=ON, yet there is no such condition 
# to keep the executables' names consistent
if(NOT MSVC)
  set_property(TARGET zfpcmd PROPERTY OUTPUT_NAME zfp)
endif()
target_link_libraries(zfpcmd zfp)
if(HAVE_LIBM_MATH)
  target_link_libraries(zfpcmd m)
endif()
