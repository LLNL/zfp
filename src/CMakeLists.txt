set(SOURCES
    bitstream.c
    decode1d.c
    decode1f.c
    decode2d.c
    decode2f.c
    decode3d.c
    decode3f.c
    encode1d.c
    encode1f.c
    encode2d.c
    encode2f.c
    encode3d.c
    encode3f.c
    zfp.c
)

if(BUILD_SHARED)
    add_library(zfp_shared SHARED ${SOURCES})

    set_target_properties(zfp_shared PROPERTIES
        OUTPUT_NAME zfp
        SOVERSION 0
        VERSION ${ZFP_VERSION_STRING}
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(zfp_shared PUBLIC ${CMAKE_SOURCE_DIR}/inc)

    install(TARGETS zfp_shared DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT LIB)
endif()

if(BUILD_STATIC)
    add_library(zfp_static STATIC ${SOURCES})

    set_target_properties(zfp_static PROPERTIES
        OUTPUT_NAME zfp
        POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(zfp_static PUBLIC ${CMAKE_SOURCE_DIR}/inc)

    if (MSVC)
        set_target_properties(zfp_static PROPERTIES PREFIX lib)
    endif()

    install(TARGETS zfp_static DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT LIB)
endif()

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/inc/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/zfp)

