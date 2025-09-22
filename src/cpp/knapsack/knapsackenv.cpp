#include "knapsackenv.hpp"

#include <algorithm>

namespace
{
    std::vector<int> extract_state(Node *node)
    {
        return node->weight;
    }
}

KnapsackEnv::KnapsackEnv()
    : reused(false),
      preprocess(false),
      method(1),
      maximization(true),
      dominance(0),
      bdd_type(0),
      maxwidth(-1),
      n_vars(0),
      n_objs(0),
      n_cons(0),
      nnds(0),
      n_comparisons(0),
      inst_kp(nullptr),
      bdd(nullptr),
      pareto_frontier(nullptr)
{
    compilation_time = timers.register_name("compilation time");
    pareto_time = timers.register_name("pareto time");
    approx_time = timers.register_name("approximation time");
    reused = true;
}

KnapsackEnv::~KnapsackEnv()
{
    clean_memory();
}

void KnapsackEnv::clean_memory()
{
    if (inst_kp != nullptr)
    {
        delete inst_kp;
        inst_kp = nullptr;
    }
    if (bdd != nullptr)
    {
        delete bdd;
        bdd = nullptr;
    }
    if (pareto_frontier != nullptr)
    {
        delete pareto_frontier;
        pareto_frontier = nullptr;
    }
}

void KnapsackEnv::initialize()
{
    if (reused)
    {
        clean_memory();
    }

    preprocess = false;
    method = 1;
    maximization = true;
    dominance = 0;
    bdd_type = 0;
    maxwidth = -1;
    order.clear();

    n_vars = 0;
    n_objs = 0;
    n_cons = 0;
    obj_coeffs.clear();
    cons_coeffs.clear();
    rhs.clear();

    initial_width = 0;
    initial_node_count = 0;
    initial_arcs_count = 0;
    reduced_width = 0;
    reduced_node_count = 0;
    reduced_arcs_count = 0;
    in_degree.clear();
    max_in_degree_per_layer.clear();
    initial_num_nodes_per_layer.clear();
    reduced_num_nodes_per_layer.clear();
    num_pareto_sol_per_layer.clear();
    num_comparisons_per_layer.clear();

    nnds = 0;
    n_comparisons = 0;
    z_sol.clear();

    timers.reset_timer(compilation_time);
    timers.reset_timer(pareto_time);
    timers.reset_timer(approx_time);
}

void KnapsackEnv::reset(bool _preprocess,
                        int _method,
                        bool _maximization,
                        int _dominance,
                        int _bdd_type,
                        int _maxwidth,
                        std::vector<int> _order)
{
    initialize();

    preprocess = _preprocess;
    method = _method;
    maximization = _maximization;
    dominance = _dominance;
    bdd_type = _bdd_type;
    maxwidth = _maxwidth;
    order = std::move(_order);
}

int KnapsackEnv::set_inst(int _n_vars,
                          int _n_cons,
                          int _n_objs,
                          std::vector<std::vector<int>> _obj_coeffs,
                          std::vector<std::vector<int>> _cons_coeffs,
                          std::vector<int> _rhs)
{
    clean_memory();

    n_vars = _n_vars;
    n_cons = _n_cons;
    n_objs = _n_objs;
    obj_coeffs = std::move(_obj_coeffs);
    cons_coeffs = std::move(_cons_coeffs);
    rhs = std::move(_rhs);

    inst_kp = new KnapsackInstance(n_vars, n_cons, n_objs, obj_coeffs, cons_coeffs, rhs);
    return 0;
}

int KnapsackEnv::preprocess_inst()
{
    if (inst_kp == nullptr)
    {
        std::cerr << "Knapsack instance not set." << std::endl;
        return 1;
    }

    timers.start_timer(compilation_time);
    if (!order.empty())
    {
        inst_kp->reset_order(order);
    }
    timers.end_timer(compilation_time);
    return 0;
}

int KnapsackEnv::initialize_dd_constructor()
{
    if (inst_kp == nullptr)
    {
        std::cerr << "Knapsack instance not set." << std::endl;
        return 1;
    }

    timers.start_timer(compilation_time);
    kp_bdd_constructor = KnapsackBDDConstructor(inst_kp);
    bdd = kp_bdd_constructor.bdd;
    timers.end_timer(compilation_time);
    return 0;
}

void KnapsackEnv::calculate_bdd_topology_stats(bool is_non_reduced)
{
    if (bdd == nullptr)
    {
        return;
    }

    if (is_non_reduced)
    {
        initial_width = bdd->get_width();
        initial_node_count = bdd->get_num_nodes();
        initial_arcs_count = bdd->get_arcs_count();
        initial_num_nodes_per_layer = bdd->get_num_nodes_per_layer();
    }
    else
    {
        reduced_width = bdd->get_width();
        reduced_node_count = bdd->get_num_nodes();
        reduced_arcs_count = bdd->get_arcs_count();
        reduced_num_nodes_per_layer = bdd->get_num_nodes_per_layer();
        in_degree = bdd->get_in_degree();
        max_in_degree_per_layer = bdd->get_max_in_degree_per_layer();
    }
}

int KnapsackEnv::generate_dd()
{
    if (bdd == nullptr)
    {
        std::cerr << "Knapsack BDD constructor not initialized." << std::endl;
        return 1;
    }

    timers.start_timer(compilation_time);
    kp_bdd_constructor.generate_exact();
    timers.end_timer(compilation_time);
    calculate_bdd_topology_stats(true);
    return 0;
}

bool KnapsackEnv::generate_next_layer()
{
    if (bdd == nullptr)
    {
        std::cerr << "Knapsack BDD constructor not initialized." << std::endl;
        return false;
    }

    timers.start_timer(compilation_time);
    bool is_done = kp_bdd_constructor.generate_next_layer();
    timers.end_timer(compilation_time);

    if (is_done)
    {
        calculate_bdd_topology_stats(true);
    }

    return is_done;
}

int KnapsackEnv::restrict_layer(int layer, std::vector<int> states_to_remove)
{
    if (bdd == nullptr || layer < 0 || layer >= static_cast<int>(bdd->layers.size()))
    {
        return -1;
    }

    if (states_to_remove.empty())
    {
        return 0;
    }

    if (states_to_remove.size() >= bdd->layers[layer].size())
    {
        return -1;
    }

    std::vector<int>::iterator it;
    std::vector<Node *> restricted_layer;
    restricted_layer.reserve(bdd->layers[layer].size() - states_to_remove.size());

    for (int i = 0; i < static_cast<int>(bdd->layers[layer].size()); ++i)
    {
        it = std::find(states_to_remove.begin(), states_to_remove.end(), i);
        if (it != states_to_remove.end())
        {
            bdd->remove_node_ref_prev(bdd->layers[layer][i]);
            states_to_remove.erase(it);
        }
        else
        {
            restricted_layer.push_back(bdd->layers[layer][i]);
        }
    }

    bdd->layers[layer] = restricted_layer;
    bdd->fix_indices(layer);
    kp_bdd_constructor.fix_state_map();

    return 0;
}

int KnapsackEnv::approximate_layer(int layer,
                                   int approx_type,
                                   std::vector<int> states_to_process)
{
    if (approx_type == 1)
    {
        return restrict_layer(layer, std::move(states_to_process));
    }

    return -1;
}

void KnapsackEnv::restrict(std::vector<std::vector<int>> states_to_remove)
{
    if (bdd == nullptr)
    {
        return;
    }

    for (int layer = 0; layer < static_cast<int>(states_to_remove.size()); ++layer)
    {
        restrict_layer(layer, states_to_remove[layer]);
    }
}

int KnapsackEnv::reduce_dd()
{
    if (bdd == nullptr)
    {
        return 1;
    }

    timers.start_timer(compilation_time);
    BDDAlg::reduce(bdd);
    timers.end_timer(compilation_time);
    calculate_bdd_topology_stats(false);
    return 0;
}

int KnapsackEnv::compute_pareto_frontier()
{
    if (bdd == nullptr)
    {
        std::cerr << "BDD not constructed! Cannot compute Pareto frontier." << std::endl;
        return 1;
    }

    MultiObjectiveStats statsMultiObj;
    pareto_frontier = nullptr;

    timers.start_timer(pareto_time);
    if (method == 1)
    {
        pareto_frontier = BDDMultiObj::pareto_frontier_topdown(bdd, maximization, 1, dominance, &statsMultiObj);
    }
    else if (method == 3)
    {
        // -- Dynamic layer cutset -- (knapsack problem type = 1)
        pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(bdd, maximization, 1, dominance, &statsMultiObj);
    }
    else
    {
        timers.end_timer(pareto_time);
        std::cerr << "Selected Pareto frontier method not supported." << std::endl;
        return 1;
    }

    timers.end_timer(pareto_time);

    if (pareto_frontier == nullptr)
    {
        std::cerr << "Pareto frontier not computed." << std::endl;
        return 1;
    }

    n_comparisons = statsMultiObj.num_comparisons;
    return 0;
}

std::vector<std::vector<int>> KnapsackEnv::get_layer(int layer)
{
    std::vector<std::vector<int>> extracted_layer;
    if (bdd == nullptr || layer < 0 || layer >= static_cast<int>(bdd->layers.size()))
    {
        return extracted_layer;
    }

    extracted_layer.reserve(bdd->layers[layer].size());
    for (Node *node : bdd->layers[layer])
    {
        extracted_layer.push_back(extract_state(node));
    }

    return extracted_layer;
}

std::vector<std::vector<std::vector<int>>> KnapsackEnv::get_dd()
{
    std::vector<std::vector<std::vector<int>>> dd;
    if (bdd == nullptr)
    {
        return dd;
    }

    if (bdd->num_layers <= 2)
    {
        return dd;
    }

    dd.resize(bdd->num_layers - 2);
    for (int layer = 1; layer < bdd->num_layers - 1; ++layer)
    {
        dd[layer - 1] = get_layer(layer);
    }

    return dd;
}

std::map<std::string, std::vector<std::vector<int>>> KnapsackEnv::get_frontier()
{
    if (pareto_frontier == nullptr)
    {
        return {};
    }
    return pareto_frontier->get_frontier();
}

double KnapsackEnv::get_time(int time_type)
{
    if (time_type == 1)
    {
        return timers.get_time(compilation_time);
    }
    if (time_type == 2)
    {
        return timers.get_time(pareto_time);
    }
    if (time_type == 3)
    {
        return timers.get_time(approx_time);
    }
    return -1.0;
}

int KnapsackEnv::get_num_nodes_per_layer(int layer)
{
    if (bdd == nullptr || layer < 0 || layer >= static_cast<int>(bdd->layers.size()))
    {
        return -1;
    }
    return static_cast<int>(bdd->layers[layer].size());
}
