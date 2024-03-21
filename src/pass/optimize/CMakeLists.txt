project(pass_optimize)

aux_source_directory(. PASS_SRC_FILES)

add_library(optimize SHARED ${PASS_SRC_FILES})