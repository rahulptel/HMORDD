#include "knapsackenv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(libknapsackenv, m)
{
    py::class_<KnapsackEnv>(m, "KnapsackEnv", py::module_local())
        .def(py::init<>())
        .def("reset", &KnapsackEnv::reset)
        .def("set_inst", &KnapsackEnv::set_inst)
        .def("preprocess_inst", &KnapsackEnv::preprocess_inst)
        .def("initialize_dd_constructor", &KnapsackEnv::initialize_dd_constructor)
        .def("generate_dd", &KnapsackEnv::generate_dd)
        .def("generate_next_layer", &KnapsackEnv::generate_next_layer)
        .def("approximate_layer", &KnapsackEnv::approximate_layer)
        .def("restrict", &KnapsackEnv::restrict)
        .def("reduce_dd", &KnapsackEnv::reduce_dd)
        .def("compute_pareto_frontier", &KnapsackEnv::compute_pareto_frontier)
        .def("get_dd", &KnapsackEnv::get_dd)
        .def("get_layer", &KnapsackEnv::get_layer)
        .def("get_frontier", &KnapsackEnv::get_frontier)
        .def("get_time", &KnapsackEnv::get_time)
        .def("get_num_nodes_per_layer", &KnapsackEnv::get_num_nodes_per_layer)
        .def_readwrite("initial_width", &KnapsackEnv::initial_width)
        .def_readwrite("initial_node_count", &KnapsackEnv::initial_node_count)
        .def_readwrite("initial_arcs_count", &KnapsackEnv::initial_arcs_count)
        .def_readwrite("reduced_width", &KnapsackEnv::reduced_width)
        .def_readwrite("reduced_node_count", &KnapsackEnv::reduced_node_count)
        .def_readwrite("reduced_arcs_count", &KnapsackEnv::reduced_arcs_count)
        .def_readwrite("max_in_degree_per_layer", &KnapsackEnv::max_in_degree_per_layer)
        .def_readwrite("initial_num_nodes_per_layer", &KnapsackEnv::initial_num_nodes_per_layer)
        .def_readwrite("reduced_num_nodes_per_layer", &KnapsackEnv::reduced_num_nodes_per_layer)
        .def_readwrite("num_pareto_sol_per_layer", &KnapsackEnv::num_pareto_sol_per_layer)
        .def_readwrite("num_comparisons_per_layer", &KnapsackEnv::num_comparisons_per_layer)
        .def_readwrite("in_degree", &KnapsackEnv::in_degree)
        .def_readwrite("nnds", &KnapsackEnv::nnds)
        .def_readwrite("z_sol", &KnapsackEnv::z_sol);
}
