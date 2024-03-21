project(pass_analysis)

aux_source_directory(. PASS_SRC_FILES)

add_library(analysis SHARED ${PASS_SRC_FILES})

target_include_directories(analysis PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../)

