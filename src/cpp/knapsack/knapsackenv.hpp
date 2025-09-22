#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "bdd/bdd.hpp"
#include "bdd/bdd_alg.hpp"
#include "bdd/bdd_multiobj.hpp"
#include "bdd/knapsack_bdd.hpp"
#include "bdd/pareto_frontier.hpp"
#include "instances/knapsack_instance.hpp"
#include "../common/util/solution.hpp"
#include "../common/util/stats.hpp"
#include "../common/util/util.hpp"

class KnapsackEnv
{
public:
    bool reused;

    // ----------------------------------------------------------------
    // Run parameters
    bool preprocess;
    int method;
    bool maximization;
    int dominance;
    int bdd_type;
    int maxwidth;
    std::vector<int> order;

    // ----------------------------------------------------------------
    // Instance data
    int n_vars;
    int n_objs;
    int n_cons;
    std::vector<std::vector<int>> obj_coeffs;
    std::vector<std::vector<int>> cons_coeffs;
    std::vector<int> rhs;

    // ----------------------------------------------------------------
    // BDD topology data
    size_t initial_width;
    size_t initial_node_count;
    size_t initial_arcs_count;
    size_t reduced_width;
    size_t reduced_node_count;
    size_t reduced_arcs_count;
    std::vector<size_t> in_degree;
    std::vector<size_t> max_in_degree_per_layer;
    std::vector<size_t> initial_num_nodes_per_layer;
    std::vector<size_t> reduced_num_nodes_per_layer;
    std::vector<size_t> num_pareto_sol_per_layer;
    std::vector<size_t> num_comparisons_per_layer;

    // ----------------------------------------------------------------
    // Pareto frontier data
    size_t nnds;
    size_t n_comparisons;
    std::vector<std::vector<int>> z_sol;

    // For statistical analysis
    Stats timers;
    int compilation_time;
    int pareto_time;
    int approx_time;

    KnapsackEnv();
    ~KnapsackEnv();

    void reset(bool preprocess,
               int method,
               bool maximization,
               int dominance,
               int bdd_type,
               int maxwidth,
               std::vector<int> order);

    int set_inst(int n_vars,
                 int n_cons,
                 int n_objs,
                 std::vector<std::vector<int>> obj_coeffs,
                 std::vector<std::vector<int>> cons_coeffs,
                 std::vector<int> rhs);

    int preprocess_inst();

    int initialize_dd_constructor();

    int generate_dd();

    bool generate_next_layer();

    int approximate_layer(int layer,
                          int approx_type,
                          std::vector<int> states_to_process);

    void restrict(std::vector<std::vector<int>> states_to_remove);

    int reduce_dd();

    int compute_pareto_frontier();

    std::vector<std::vector<int>> get_layer(int layer);

    std::vector<std::vector<std::vector<int>>> get_dd();

    std::map<std::string, std::vector<std::vector<int>>> get_frontier();

    double get_time(int time_type);

    int get_num_nodes_per_layer(int layer);

private:
    void initialize();
    void clean_memory();
    void calculate_bdd_topology_stats(bool is_non_reduced);
    int restrict_layer(int layer, std::vector<int> states_to_remove);

    KnapsackInstance *inst_kp;
    KnapsackBDDConstructor kp_bdd_constructor;
    BDD *bdd;
    ParetoFrontier *pareto_frontier;
};
