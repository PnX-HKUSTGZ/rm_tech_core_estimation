cmake_minimum_required(VERSION 3.16)
project(tech_core_estimation LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV 4.5.3 REQUIRED COMPONENTS core calib3d)

add_library(${PROJECT_NAME} SHARED src/pnp_estimator.cpp src/candidates.cpp src/refine.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen PRIVATE opencv_core opencv_calib3d)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(yaml-cpp REQUIRED)
  ament_add_gtest(test_pnp_estimator test/test_pnp_estimator.cpp)
  target_link_libraries(test_pnp_estimator ${PROJECT_NAME})
  ament_add_gtest(test_reference_cases test/test_reference_cases.cpp)
  target_link_libraries(test_reference_cases ${PROJECT_NAME} yaml-cpp::yaml-cpp)
  target_compile_definitions(test_reference_cases PRIVATE
    PNP_CASES_FILE="${CMAKE_CURRENT_SOURCE_DIR}/test/data/pnp_cases.yaml")
endif()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)
ament_package()
