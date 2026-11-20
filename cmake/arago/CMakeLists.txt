# The language frontend uses ONNX headers even for trace-driven workloads.
# The target device owns its transport, so this build omits legacy NoC/DRAM libraries.
include(FetchContent)
find_package(nlohmann_json 3.11 REQUIRED)
find_path(ONNX_INCLUDE_DIR onnx/defs/schema.h REQUIRED)
find_path(PROTOBUF_INCLUDE_DIR google/protobuf/message.h REQUIRED)
find_library(ONNX_LIBRARY onnx REQUIRED)
find_library(ONNX_PROTO_LIBRARY onnx_proto REQUIRED)
find_library(PROTOBUF_LIBRARY protobuf REQUIRED)
find_path(BOOST_INCLUDE_DIR boost/program_options.hpp REQUIRED)
find_library(BOOST_PROGRAM_OPTIONS boost_program_options REQUIRED)

FetchContent_Declare(spdlog URL https://codeload.github.com/gabime/spdlog/tar.gz/ad0e89cbfb4d0c1ce4d097e134eb7be67baebb36
  URL_HASH SHA256=3430bf46c205be0022e5f66c578b91c7f274ca9ce171256494223cca9102e22a)
FetchContent_Declare(robinhood URL https://codeload.github.com/martinus/robin-hood-hashing/tar.gz/9145f963d80d6a02f0f96a47758050a89184a3ed
  URL_HASH SHA256=2ec51a99f9446cb60843391e05c9bf8a1e040a7f5e67e5db9e78fc9e9dc87f9b)
FetchContent_MakeAvailable(spdlog robinhood)

set(src "${PROJECT_SOURCE_DIR}/src")
file(GLOB operators CONFIGURE_DEPENDS "${src}/operations/*.cc")
add_library(onnxim_language STATIC
  ${src}/Common.cc ${src}/Mapping.cc ${src}/Tensor.cc ${src}/Model.cc
  ${src}/models/LanguageModel.cc ${src}/scheduler/LanguageScheduler.cc
  ${src}/scheduler/IterLevelScheduler.cc ${operators})
target_compile_features(onnxim_language PUBLIC cxx_std_20)
target_compile_definitions(onnxim_language PUBLIC ONNX_ML=1 ONNX_NAMESPACE=onnx)
target_include_directories(onnxim_language PUBLIC ${src} ${ONNX_INCLUDE_DIR} ${PROTOBUF_INCLUDE_DIR}
  "${robinhood_SOURCE_DIR}/src/include")
target_link_libraries(onnxim_language PUBLIC spdlog::spdlog nlohmann_json::nlohmann_json
  ${ONNX_LIBRARY} ${ONNX_PROTO_LIBRARY} ${PROTOBUF_LIBRARY})
target_precompile_headers(onnxim_language PRIVATE "${src}/Common.h")

add_subdirectory("${src}/arago" device)
# Preserve the focused build's public artifact path alongside the CLI.
set_target_properties(arago_device PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
add_executable(onnxim-arago ${src}/main.cc ${src}/helper/CommandLineParser.cc ${src}/arago/Run.cc)
target_compile_definitions(onnxim-arago PRIVATE ONNXIM_ARAGO_ONLY)
target_include_directories(onnxim-arago PRIVATE ${BOOST_INCLUDE_DIR})
target_link_libraries(onnxim-arago PRIVATE ${BOOST_PROGRAM_OPTIONS})
target_link_libraries(onnxim-arago PRIVATE arago_device onnxim_language)
foreach(test IN ITEMS hardware layer)
  add_executable(arago-test-${test} ${PROJECT_SOURCE_DIR}/tests/arago/test_${test}.cc)
  target_link_libraries(arago-test-${test} PRIVATE arago_device)
  add_test(NAME arago-${test} COMMAND arago-test-${test})
endforeach()

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME arago-language-integration COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/tools/arago/run_comparison.py --simulator $<TARGET_FILE:onnxim-arago>)
add_test(NAME arago-batch16-flashattention COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/tools/arago/run_comparison.py --simulator $<TARGET_FILE:onnxim-arago> --workload batch16)
add_test(NAME arago-architecture-contracts COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/tests/arago/test_architecture.py $<TARGET_FILE:onnxim-arago>)
set(ARAGO_TORCH_PYTHON "" CACHE FILEPATH "Python interpreter with PyTorch for the numerical contracts")
if(ARAGO_TORCH_PYTHON)
  add_test(NAME arago-precision-contracts COMMAND ${ARAGO_TORCH_PYTHON}
    ${PROJECT_SOURCE_DIR}/tests/arago/test_precision.py cpu)
endif()
if(ARAGO_TORCH_PYTHON)
  add_test(NAME arago-panel-replay COMMAND ${ARAGO_TORCH_PYTHON}
    ${PROJECT_SOURCE_DIR}/tools/arago/replay_panels.py --simulator $<TARGET_FILE:onnxim-arago>)
endif()
add_test(NAME arago-case-elasticity COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/tests/arago/test_case_elasticity.py $<TARGET_FILE:onnxim-arago>)
add_test(NAME arago-four-case-metrics COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/tests/arago/test_four_case_metrics.py)
add_test(NAME arago-interactivity-recommendations COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/tests/arago/test_recommendation_study.py)
add_test(NAME arago-report-cases COMMAND ${Python3_EXECUTABLE}
  ${PROJECT_SOURCE_DIR}/tests/arago/test_report_cases.py)
