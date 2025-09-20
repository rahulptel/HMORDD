// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

// General includes
#include <iostream>
#include <cstdlib>

#include "../common/bdd/bdd.hpp"
#include "../common/bdd/bdd_alg.hpp"
#include "../common/bdd/bdd_multiobj.hpp"
#include "../common/util/stats.hpp"
#include "../common/util/util.hpp"
#include "../common/bdd/pareto_frontier.hpp"

// Set packing / Independent set includes
#include "instances/indepset_instance.hpp"
#include "instances/setpacking_instance.hpp"
#include "bdd/indepset_bdd.hpp"

class SetpackingEnv
{
public:
    bool reused;

    // ----------------------------------------------------------------
    // Run parameters
    int method;
    bool maximization;
    int bdd_type;
    int maxwidth;
    vector<int> order;

    // ----------------------------------------------------------------
    // Instance data
    int n_vars;
    int n_objs;
    int n_cons;
    vector<vector<int>> obj_coeffs;
    vector<vector<int>> cons_coeffs;
    vector<int> rhs;

    // ----------------------------------------------------------------
    // Objective coefficients after static/dynamic reordering
    vector<vector<int>> obj_coefficients;

    // ----------------------------------------------------------------
    // BDD topology data

    size_t initial_width, initial_node_count, initial_arcs_count;
    size_t reduced_width, reduced_node_count, reduced_arcs_count;
    vector<size_t> in_degree, max_in_degree_per_layer;
    vector<size_t> initial_num_nodes_per_layer, reduced_num_nodes_per_layer;
    vector<size_t> num_pareto_sol_per_layer;
    vector<size_t> num_comparisons_per_layer;

    // ----------------------------------------------------------------
    // Pareto frontier data
    size_t nnds = 0;
    size_t n_comparisons = 0;
    vector<vector<int>> z_sol;

    // For statistical analysis
    // ----------------------------------------------------------------
    Stats timers;
    int compilation_time = timers.register_name("compilation time");
    int pareto_time = timers.register_name("pareto time");
    int approx_time = timers.register_name("approximation time");

    SetpackingEnv();

    ~SetpackingEnv();

    void reset();

    int set_inst(int n_vars,
                 int n_cons,
                 int n_objs,
                 vector<vector<int>> obj_coeffs,
                 vector<vector<int>> cons_coeffs);

    int initialize_dd_constructor();
    void set_var_layer(int l);
    int generate_next_layer();
    void generate_exact_dd();
    void generate_restricted_dd(int, int);
    // void calculate_bdd_topology_stats(bool is_non_reduced);
    vector<vector<int>> get_layer(int);
    size_t get_width();
    vector<vector<vector<int>>> get_dd();
    vector<int> get_var_layer();
    int get_num_nodes_per_layer(int);
    size_t count_incoming_arcs(Node *node);
    size_t get_total_incoming_arcs_count();
    size_t get_total_nodes_count();

    vector<vector<int>> get_last_built_layer();
    vector<int> get_last_layer_state();
    int get_num_nodes_per_last_layer();

    void compute_pareto_frontier(int);
    vector<int> get_frontier();
    double get_pareto_time();
    double get_compilation_time();

    BDD *bdd;
    ParetoFrontier *pareto_frontier;
    // private:
    void initialize();
    void clean_memory();

    // ----------------------------------------------------------------
    // Instances
    SetPackingInstance inst_setpack;
    IndepSetInst *inst_indepset;

    // ----------------------------------------------------------------
    // DD constructor
    IndepSetBDDConstructor indset_bdd_constructor;

    // ----------------------------------------------------------------
    // DD
};