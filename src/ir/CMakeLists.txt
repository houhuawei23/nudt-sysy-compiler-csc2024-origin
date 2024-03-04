
project(ir)
aux_source_directory(. IR_SRC_FILES)

add_library(lib_ir
    SHARED ${IR_SRC_FILES}
)

target_include_directories(lib_ir INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# target_link_directories(irlib
#     INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/)