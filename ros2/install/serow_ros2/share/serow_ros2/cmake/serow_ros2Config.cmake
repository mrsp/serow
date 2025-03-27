# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_serow_ros2_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED serow_ros2_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(serow_ros2_FOUND FALSE)
  elseif(NOT serow_ros2_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(serow_ros2_FOUND FALSE)
  endif()
  return()
endif()
set(_serow_ros2_CONFIG_INCLUDED TRUE)

# output package information
if(NOT serow_ros2_FIND_QUIETLY)
  message(STATUS "Found serow_ros2: 1.0.0 (${serow_ros2_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'serow_ros2' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${serow_ros2_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(serow_ros2_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "ament_cmake_export_dependencies-extras.cmake")
foreach(_extra ${_extras})
  include("${serow_ros2_DIR}/${_extra}")
endforeach()
