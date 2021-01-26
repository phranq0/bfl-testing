cmake_minimum_required(VERSION 3.10)
project(nonlinear_particle_filter)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find bfl - always use this if a package is installed within ros
find_package(PkgConfig)
pkg_check_modules(BFL REQUIRED orocos-bfl)
include_directories(${BFL_INCLUDE_DIRS}/bfl)
include_directories(./)
include_directories(./Mobile_Robot)

link_directories(/opt/ros/melodic/lib)

add_executable( test_nonlinear_particle main_particleFilter.cpp nonlinearSystemPdf.cpp nonlinearMeasurementPdf.cpp ./Mobile_Robot/mobile_robot.cpp ./Mobile_Robot/nonlinearanalyticconditionalgaussianmobile.cpp) 
target_link_libraries( test_nonlinear_particle orocos-bfl ${MATRIX_LIBS} ${RNG_LIBS}  ${GINAC_LIBS})