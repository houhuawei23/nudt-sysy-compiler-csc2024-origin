
# add_executable(sysyc
#     sysyc.cpp
#     ASTPrinter.cpp)

# target_include_directories(sysyc PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/antlr4)
# target_link_libraries(sysyc PRIVATE SysYParser)

# set(THREADS_PREFER_PTHREAD_FLAG ON)
# find_package(Threads REQUIRED)
# target_link_libraries(sysyc PRIVATE Threads::Threads)

## main.cpp


add_executable(main
    main.cpp
)
# antlr4 include
target_include_directories(main PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/antlr4)
target_link_libraries(main PRIVATE SysYParser)
# visitor include
target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/visitor)
target_link_libraries(main PRIVATE lib_visitor)



add_subdirectory(antlr4)
add_subdirectory(visitor)
add_subdirectory(ir)
