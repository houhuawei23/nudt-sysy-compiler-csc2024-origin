
add_executable(sysyc
    sysyc.cpp
    ASTPrinter.cpp)

target_include_directories(sysyc PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/antlr4)
target_link_libraries(sysyc PRIVATE SysYParser)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(sysyc PRIVATE Threads::Threads)

add_subdirectory(antlr4)