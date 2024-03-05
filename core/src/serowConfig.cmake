add_library(serow STATIC IMPORTED)
find_library(SEROW_LIBRARY_PATH serow HINTS "${CMAKE_CURRENT_LIST_DIR}/../../")
set_target_properties(serow PROPERTIES IMPORTED_LOCATION "${SEROW_LIBRARY_PATH}")
