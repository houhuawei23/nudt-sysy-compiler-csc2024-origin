
project(ir)
aux_source_directory(. IR_SRC_FILES)

add_library(irlib
    SHARED ${IR_SRC_FILES}
)

target_include_directories(irlib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# target_link_directories(irlib
#     INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/)