project(generic)
aux_source_directory(. GENERIC_SRC_FILES)

add_library(generic
    SHARED ${GENERIC_SRC_FILES}
)
