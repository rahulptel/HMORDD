#include "setpackingenv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(libsetpackingenv, m)
{
    py::class_<SetpackingEnv>(m, "SetpackingEnv", py::module_local())
        .def(py::init<>())
        .def("reset", &SetpackingEnv::reset)
        .def("set_inst", &SetpackingEnv::set_inst)
        .def("initialize_dd_constructor", &SetpackingEnv::initialize_dd_constructor)
        .def("generate_next_layer", &SetpackingEnv::generate_next_layer)
        .def("generate_exact_dd", &SetpackingEnv::generate_exact_dd)
        .def("generate_restricted_dd", &SetpackingEnv::generate_restricted_dd)
        .def("get_layer", &SetpackingEnv::get_layer)
        .def("get_dd", &SetpackingEnv::get_dd)
        .def("get_width", &SetpackingEnv::get_width)
        .def("set_var_layer", &SetpackingEnv::set_var_layer)
        .def("get_var_layer", &SetpackingEnv::get_var_layer)
        .def("get_num_nodes_per_layer", &SetpackingEnv::get_num_nodes_per_layer)
        .def("get_last_built_layer", &SetpackingEnv::get_last_built_layer)
        .def("get_last_layer_state", &SetpackingEnv::get_last_layer_state)
        .def("get_num_nodes_per_last_layer", &SetpackingEnv::get_num_nodes_per_last_layer)
        .def("get_total_incoming_arcs_count", &SetpackingEnv::get_total_incoming_arcs_count)
        .def("get_total_nodes_count", &SetpackingEnv::get_total_nodes_count)
        .def("compute_pareto_frontier", &SetpackingEnv::compute_pareto_frontier)
        .def("get_frontier", &SetpackingEnv::get_frontier)
        .def("get_pareto_time", &SetpackingEnv::get_pareto_time)
        .def("get_compilation_time", &SetpackingEnv::get_compilation_time);
}
