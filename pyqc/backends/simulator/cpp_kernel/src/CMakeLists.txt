include_directories(../include)

set(LIBRARY_OUTPUT_PATH ../lib)

aux_source_directory(./ DIR_SRCS)

add_library(fas SHARED ${DIR_SRCS})
