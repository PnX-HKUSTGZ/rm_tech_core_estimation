cmake_minimum_required(VERSION 3.16)
project(tech_core_estimation_model LANGUAGES CXX)

find_package(ament_cmake REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(yaml-cpp REQUIRED)

add_library(${PROJECT_NAME} SHARED src/tech_core_model.cpp src/model_meta.cpp src/image_ops.cpp)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(${PROJECT_NAME} PUBLIC opencv_core PRIVATE opencv_imgproc opencv_dnn
  openvino::runtime yaml-cpp::yaml-cpp)

add_executable(tech_core_model_benchmark tools/benchmark.cpp)
target_link_libraries(tech_core_model_benchmark PRIVATE ${PROJECT_NAME} opencv_imgcodecs yaml-cpp::yaml-cpp)

set(TECH_CORE_GOLDEN_DIR "" CACHE PATH "Model release directory for the PyTorch golden tests")
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_image_ops test/test_image_ops.cpp)
  target_include_directories(test_image_ops PRIVATE src)
  target_link_libraries(test_image_ops ${PROJECT_NAME})
  if(TECH_CORE_GOLDEN_DIR)
    ament_add_gtest(test_golden test/test_golden.cpp)
    target_compile_definitions(test_golden PRIVATE TECH_CORE_GOLDEN_DIR="${TECH_CORE_GOLDEN_DIR}")
    target_link_libraries(test_golden ${PROJECT_NAME} opencv_imgcodecs yaml-cpp::yaml-cpp)
  endif()
endif()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS ${PROJECT_NAME} EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS tech_core_model_benchmark DESTINATION lib/${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(OpenCV)
ament_package()
