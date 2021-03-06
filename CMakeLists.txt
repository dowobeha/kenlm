cmake_minimum_required(VERSION 2.8.8)
#
# The KenLM cmake files make use of add_library(... OBJECTS ...)
# 
# This syntax allows grouping of source files when compiling
# (effectively creating "fake" libraries based on source subdirs).
# 
# This syntax was only added in cmake version 2.8.8
#
# see http://www.cmake.org/Wiki/CMake/Tutorials/Object_Library


# This CMake file was created by Lane Schwartz <dowobeha@gmail.com>


# Define a single cmake project
project(kenlm)

#include(CTest)

# Compile all libraries into lib/
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Compile all libraries into lib/
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)


# Tell CMake to enable compiling KenLM unit test code
#enable_testing()

#set_property(GLOBAL PROPERTY USE_FOLDERS ON)
#FILE(GLOB all */*.*)
#source_group( foobar FILES ${all})
#set(Boost_DEBUG 1)
#source_group( lm REGULAR_EXPRESSION "${PROJECT_SOURCE_DIR}/lm/.*" )
# We need boost
find_package(Boost 1.36.0 REQUIRED COMPONENTS
  program_options 
  system
  thread
  unit_test_framework
)

MESSAGE("Boost information:") 
MESSAGE("  Boost_INCLUDE_DIR: ${Boost_INCLUDE_DIR}") 
MESSAGE("  Boost_LIBRARIES: ${Boost_LIBRARIES}") 
MESSAGE("  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}") 
message("  Boost_UNIT_TEST_FRAMEWORK_LIBRARY: ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}")

# Define where include files live
include_directories(
  ${PROJECT_SOURCE_DIR} 
  ${Boost_INCLUDE_DIR}
)


# Process subdirectories
add_subdirectory(util)
add_subdirectory(lm)

