#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../pyqc/backends/simulator/cpp_kernel/include/full_amplitude_sim.hpp"


PYBIND11_MODULE(fullAlib, m) {
    m.doc() = "pybind11 fullAlib plugin";
    pybind11::class_<fullASim>(m, "fullASim")
        .def(pybind11::init())
        .def("flush", &fullASim::flush)
        .def("applyOneGate", [](fullASim &a, pybind11::array_t<StateType> matrix, int target, int tag){
            pybind11::buffer_info buf = matrix.request();
            StateType *ptr = (StateType *) buf.ptr;
            a.applyOneGate(ptr, target, tag);
        })
        .def("applyControlOneGate", [](fullASim &a, pybind11::array_t<StateType> matrix, int target, int control, int tag){
            pybind11::buffer_info buf = matrix.request();
            StateType *ptr = (StateType *) buf.ptr;
            a.applyControlOneGate(ptr, target, control, tag);
        })
        .def("getOneAmplitudeFromBinstring",[](fullASim &a, std::string binstring){
            StateType amplitude = a.getOneAmplitudeFromBinstring(binstring);
            return amplitude;
        })
        .def("getExpectation", [](fullASim &a, pybind11::array_t<int> target, int size){
            pybind11::buffer_info buf = target.request();
            int *ptr = (int *) buf.ptr;
            return a.getExpectation(ptr, size);
        })
        .def("grad_helper_init", [](fullASim &a, pybind11::array_t<StateType> matrix_list, pybind11::array_t<int>target_list, pybind11::array_t<int>size_list, int size){
            pybind11::buffer_info matrix_buf = matrix_list.request();
            pybind11::buffer_info target_buf = target_list.request();
            pybind11::buffer_info size_buf = size_list.request();
            StateType *matrix_ptr = (StateType *) matrix_buf.ptr;
            int *target_ptr = (int *) target_buf.ptr;
            int *size_ptr = (int *) size_buf.ptr;
            a.grad_helper_init(matrix_ptr, target_ptr, size_ptr, size);
        })
        .def("grad_helper", [](fullASim &a, pybind11::array_t<StateType> matrix_list, pybind11::array_t<int>target_list, pybind11::array_t<int>size_list, int size){
            pybind11::buffer_info matrix_buf = matrix_list.request();
            pybind11::buffer_info target_buf = target_list.request();
            pybind11::buffer_info size_buf = size_list.request();
            StateType *matrix_ptr = (StateType *) matrix_buf.ptr;
            int *target_ptr = (int *) target_buf.ptr;
            int *size_ptr = (int *) size_buf.ptr;
            return a.grad_helper(matrix_ptr, target_ptr, size_ptr, size);
        })
        .def("show_state", &fullASim::show_state);
}