cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

message(VERBOSE "Executing update step for core")

block(SCOPE_FOR VARIABLES)

include("/workspace/build/CMakeFiles/fc-tmp/core/core-gitupdate.cmake")

endblock()
