#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "PQP::PQP" for configuration "MinSizeRel"
set_property(TARGET PQP::PQP APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(PQP::PQP PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/PQP.lib"
  )

list(APPEND _cmake_import_check_targets PQP::PQP )
list(APPEND _cmake_import_check_files_for_PQP::PQP "${_IMPORT_PREFIX}/lib/PQP.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
