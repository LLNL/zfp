# if(ZFP_WITH_HIP)
  # set_source_files_properties(${zfp_source} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  # hip_add_executable(zfpcmd zfp.c)

# else()
	add_executable(zfpcmd zfp.c)
# endif()

# protect against LNK1114: cannot overwrite the original file 'lib/Release/zfp.lib'; error code 32;
# rationale: linker can't handle the case of an executable file having the same name as a library file
if(NOT MSVC)
  set_property(TARGET zfpcmd PROPERTY OUTPUT_NAME zfp)
endif()

target_link_libraries(zfpcmd zfp)
if(HAVE_LIBM_MATH)
  target_link_libraries(zfpcmd m)
endif()
