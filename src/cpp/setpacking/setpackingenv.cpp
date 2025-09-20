#include "setpackingenv.hpp"

vector<int> dynamicBitsetToVector(const boost::dynamic_bitset<> &bitset)
{
    vector<int> result;

    for (boost::dynamic_bitset<>::size_type i = 0; i < bitset.size(); ++i)
    {
        if (bitset[i])
        {
            result.push_back(i);
        }
    }

    return result;
}

SetpackingEnv::SetpackingEnv()
{
    reused = false;
    initialize();
    reused = true;
}

SetpackingEnv::~SetpackingEnv()
{
    clean_memory();
}

void SetpackingEnv::clean_memory()
{
    if (inst_indepset != NULL)
    {
        delete inst_indepset;
    }
    if (bdd != NULL)
    {
        delete bdd;
    }
    if (pareto_frontier != NULL)
    {
        delete pareto_frontier;
    }
}

void SetpackingEnv::initialize()
{
    maximization = true;

    if (!reused)
    {
        inst_indepset = NULL;
        bdd = NULL;
        pareto_frontier = NULL;
    }
    else
    {
        clean_memory();
    }

    initial_width = 0, initial_node_count = 0, initial_arcs_count = 0;
    reduced_width = 0, reduced_node_count = 0, reduced_arcs_count = 0;
    in_degree.clear();
    max_in_degree_per_layer.clear();
    initial_num_nodes_per_layer.clear();
    reduced_num_nodes_per_layer.clear();

    // timers.reset_timer(compilation_time);
    // timers.reset_timer(pareto_time);
    // timers.reset_timer(approx_time);

    nnds = 0;
    n_comparisons = 0;
    num_pareto_sol_per_layer.clear();
    z_sol.clear();
}

void SetpackingEnv::reset()
{
    initialize();
}

int SetpackingEnv::set_inst(int n_vars,
                            int n_cons,
                            int n_objs,
                            vector<vector<int>> obj_coeffs,
                            vector<vector<int>> cons_coeffs)
{
    // cout << "Setting set packing problem..." << endl;
    // _cons_coeff will have variable per constraint array of the shape n_cons x n_vars in constraint
    // variable should be indexed starting from 0
    inst_setpack = SetPackingInstance(n_vars, n_cons, n_objs, obj_coeffs, cons_coeffs);

    // create associated independent set instance
    inst_indepset = inst_setpack.create_indepset_instance();
    // inst_indepset = new IndepSetInst(n_vars, cons_coeffs, obj_coeffs);
    inst_indepset->obj_coeffs = obj_coeffs;

    return 0;
}

int SetpackingEnv::initialize_dd_constructor()
{
    timers.start_timer(compilation_time);

    // generate independent set BDD
    indset_bdd_constructor = IndepSetBDDConstructor(inst_indepset, inst_indepset->obj_coeffs);

    // indset_bdd_constructor.var_layer.clear();
    bdd = indset_bdd_constructor.bdd;

    timers.end_timer(compilation_time);

    return 0;
}

void SetpackingEnv::set_var_layer(int v)
{
    indset_bdd_constructor.set_var_layer(v);
}

// void SetpackingEnv::calculate_bdd_topology_stats(bool is_non_reduced)
// {
//     if (is_non_reduced)
//     {
//         initial_width = bdd->get_width();
//         initial_node_count = bdd->get_num_nodes();
//         initial_arcs_count = bdd->get_arcs_count();
//         initial_num_nodes_per_layer = bdd->get_num_nodes_per_layer();
//     }
//     else
//     {
//         reduced_width = bdd->get_width();
//         reduced_node_count = bdd->get_num_nodes();
//         reduced_arcs_count = bdd->get_arcs_count();
//         reduced_num_nodes_per_layer = bdd->get_num_nodes_per_layer();

//         in_degree = bdd->get_in_degree();
//         max_in_degree_per_layer = bdd->get_max_in_degree_per_layer();
//     }
// }

int SetpackingEnv::generate_next_layer()
{
    bool is_done = indset_bdd_constructor.generate_next_layer();

    return is_done;
}

vector<vector<int>> SetpackingEnv::get_layer(int l)
{
    vector<vector<int>> layer;
    vector<int> zp, op;
    layer.reserve(bdd->layers[l].size());

    for (vector<Node *>::iterator it = bdd->layers[l].begin();
         it != bdd->layers[l].end(); ++it)
    {
        // zp.clear();
        // op.clear();

        // // Indices of zero prev
        // for (vector<Node *>::iterator it1 = (*it)->prev[0].begin(); it1 != (*it)->prev[0].end(); ++it1)
        // {
        //     zp.push_back((*it1)->index);
        // }

        // // Indices of one prev

        // for (vector<Node *>::iterator it1 = (*it)->prev[1].begin(); it1 != (*it)->prev[1].end(); ++it1)
        // {
        //     op.push_back((*it1)->index);
        // }

        // layer.push_back({{"s", dynamicBitsetToVector((*it)->setpack_state)},
        //                     {"op", op},
        //                     {"zp", zp}});

        layer.push_back(dynamicBitsetToVector((*it)->setpack_state));
    }

    return layer;
}

size_t SetpackingEnv::count_incoming_arcs(Node *node)
{
    return node->prev[0].size() + node->prev[1].size();
}

size_t SetpackingEnv::get_total_incoming_arcs_count()
{
    assert(bdd != NULL);
    return bdd->get_total_in_degree();
}

size_t SetpackingEnv::get_total_nodes_count()
{
    assert(bdd != NULL);
    return bdd->get_num_nodes();
}

vector<vector<int>> SetpackingEnv::get_last_built_layer()
{
    return get_layer(indset_bdd_constructor.l - 1);
}

vector<int> SetpackingEnv::get_last_layer_state()
{
    vector<int> last_layer_state(indset_bdd_constructor.inst->graph->n_vertices, 0);
    for (vector<Node *>::iterator it = bdd->layers[indset_bdd_constructor.l - 1].begin();
         it != bdd->layers[indset_bdd_constructor.l - 1].end();
         ++it)
    {

        for (boost::dynamic_bitset<>::size_type i = 0;
             i < (*it)->setpack_state.size();
             ++i)
        {
            if ((*it)->setpack_state[i])
            {
                ++last_layer_state[i];
            }
        }
    }

    return last_layer_state;
}

void SetpackingEnv::generate_exact_dd()
{
    indset_bdd_constructor.generate_exact();
}

void SetpackingEnv::generate_restricted_dd(int rest_width, int node_select)
{
    indset_bdd_constructor.generate_restricted_dd(rest_width, node_select);
}

int SetpackingEnv::get_num_nodes_per_last_layer()
{
    return bdd->layers[indset_bdd_constructor.l - 1].size();
}

vector<int> SetpackingEnv::get_frontier()
{
    return pareto_frontier->sols;
}

size_t SetpackingEnv::get_width()
{
    return bdd->get_width();
}

vector<vector<vector<int>>> SetpackingEnv::get_dd()
{
    vector<vector<vector<int>>> dd;
    dd.resize(bdd->num_layers - 2);

    for (int l = 1; l < bdd->num_layers - 1; ++l)
    {

        dd[l - 1] = get_layer(l);
    }

    return dd;
}

vector<int> SetpackingEnv::get_var_layer()
{
    return indset_bdd_constructor.var_layer;
}

int SetpackingEnv::get_num_nodes_per_layer(int layer)
{
    if (bdd != NULL)
    {
        return bdd->layers[layer].size();
    }
    return -1;
}

double SetpackingEnv::get_compilation_time()
{
    return timers.get_time(compilation_time);
}

void SetpackingEnv::compute_pareto_frontier(int method)
{
    MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;

    timers.start_timer(pareto_time);
    // cout << method << endl;
    if (method == 1)
    {
        // -- Optimal BFS algorithm: top-down --
        pareto_frontier = BDDMultiObj::pareto_frontier_topdown(bdd, maximization, -1, 0, statsMultiObj);
    }
    else if (method == 2)
    {
        // -- Optimal BFS algorithm: bottom-up --
        pareto_frontier = BDDMultiObj::pareto_frontier_bottomup(bdd, maximization, -1, 0, statsMultiObj);
    }
    else if (method == 3)
    {
        // -- Dynamic layer cutset --
        pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(bdd, maximization, -1, 0, statsMultiObj);
    }
    timers.end_timer(pareto_time);
    // delete statsMultiObj;
}

double SetpackingEnv::get_pareto_time()
{
    return timers.get_time(pareto_time);
}