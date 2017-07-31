include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# compile libraries used by tests
add_subdirectory(utils)

# compile tests
add_subdirectory(inline)
add_subdirectory(misc)
add_subdirectory(encode)
add_subdirectory(decode)
add_subdirectory(endtoend)
