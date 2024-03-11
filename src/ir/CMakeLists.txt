
project(ir)
aux_source_directory(. IR_SRC_FILES)

add_library(ir
    SHARED ${IR_SRC_FILES}
)

target_include_directories(ir PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# target_link_directories(irlib
#     INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/)