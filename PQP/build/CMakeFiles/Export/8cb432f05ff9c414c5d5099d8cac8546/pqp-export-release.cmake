#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "PQP::PQP" for configuration "Release"
set_property(TARGET PQP::PQP APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(PQP::PQP PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/PQP.lib"
  )

list(APPEND _cmake_import_check_targets PQP::PQP )
list(APPEND _cmake_import_check_files_for_PQP::PQP "${_IMPORT_PREFIX}/lib/PQP.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
