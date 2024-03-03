
aux_source_directory(. VISITOR_SRC_FILES)

add_library(visitor
    SHARED ${VISITOR_SRC_FILES}
)

# antlr include and link
target_include_directories(visitor PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../antlr4)
target_link_libraries(visitor PRIVATE SysYParser)

# ir link
target_link_libraries(visitor PRIVATE irlib)
# target_include_directories(visitor PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../ir/include)
# target_link_directories(visitor PUBLIC antlr4)