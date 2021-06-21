find_package(pybind11 REQUIRED)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/pyqc/backends/simulator/libs)

pybind11_add_module(fullAlib SHARED fullSim_wapper.cpp)

target_link_libraries(fullAlib PRIVATE fas)