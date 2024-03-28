
project(mir)

aux_source_directory(. IR_SRC_FILES)

add_library(mir
    SHARED ${IR_SRC_FILES}
)

