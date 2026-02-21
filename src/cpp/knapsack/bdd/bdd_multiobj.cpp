// ----------------------------------------------------------
// BDD Multiobjective Algorithms - Implementation
// ----------------------------------------------------------

#include "bdd_multiobj.hpp"
#include "bdd_alg.hpp"

typedef std::pair<int, int> intpair;

inline bool IntPairLargestToSmallestComp(intpair l, intpair r)
{
	return l.second > r.second; // from largest to smallest
}

inline bool SetPackingStateMinElementSmallestToLargestComp(Node *l, Node *r)
{
	return l->setpack_state.find_first() < r->setpack_state.find_first(); // from smallest to largest
}

//
// Find pareto frontier using top-down approach
//
ParetoFrontier *BDDMultiObj::pareto_frontier_topdown(BDD *bdd, bool maximization, const int problem_type, int dominance_strategy, MultiObjectiveStats *stats)
{
	// cout << "\nComputing Pareto Frontier..." << endl;

	// Initialize stats
	stats->pareto_dominance_time = 0;
	stats->pareto_dominance_filtered = 0;
	stats->num_comparisons = 0;
	clock_t time_filter = 0, init;

	// Initialize manager
	ParetoFrontierManager *mgmr = new ParetoFrontierManager(bdd->get_width());

	// Create root solution
	// list<int> x;
	// ObjType zero_array[NOBJS];
	// memset(zero_array, 0, sizeof(ObjType) * NOBJS);
	// Solution root_sol(x, zero_array);
	// // Set root PF
	// bdd->get_root()->pareto_frontier = mgmr->request();
	// bdd->get_root()->pareto_frontier->add(root_sol);
	vector<int> x;
	vector<int> obj(NOBJS, 0);
	Solution rootSol(x, obj);
	bdd->get_root()->pareto_frontier = mgmr->request();
	bdd->get_root()->pareto_frontier->add(rootSol);

	vector<float> avg_sols_per_node;
	int total_sol_per_layer;

	if (maximization)
	{
		for (int l = 1; l < bdd->num_layers; ++l)
		{
			// cout << "\tLayer " << l << " - size = " << bdd->layers[l].size() << '\n';

			// iterate on layers
			total_sol_per_layer = 0;
			for (vector<Node *>::iterator it = bdd->layers[l].begin(); it != bdd->layers[l].end(); ++it)
			{
				Node *node = (*it);
				int id = node->index;

				// Request frontier
				node->pareto_frontier = mgmr->request();

				// add incoming one arcs
				for (vector<Node *>::iterator prev = (*it)->prev[1].begin();
					 prev != (*it)->prev[1].end(); ++prev)
				{
					stats->num_comparisons += node->pareto_frontier->merge(*((*prev)->pareto_frontier),
																		   1,
																		   (*prev)->weights[1]);
				}

				// add incoming zero arcs
				for (vector<Node *>::iterator prev = (*it)->prev[0].begin();
					 prev != (*it)->prev[0].end(); ++prev)
				{
					stats->num_comparisons += node->pareto_frontier->merge(*((*prev)->pareto_frontier),
																		   0,
																		   (*prev)->weights[0]);
				}
				total_sol_per_layer += (node->pareto_frontier->sols.size());
			}
			// cout << l << ": " << total_sol_per_layer << " " << bdd->layers[l].size() << " " << total_sol_per_layer / bdd->layers[l].size() << endl;

			if (dominance_strategy > 0)
			{
				init = clock();
				BDDMultiObj::filter_dominance(bdd, l, problem_type, dominance_strategy, stats);
				stats->pareto_dominance_time += clock() - init;
			}

			// Deallocate frontier from previous layer
			for (vector<Node *>::iterator it = bdd->layers[l - 1].begin(); it != bdd->layers[l - 1].end(); ++it)
			{
				mgmr->deallocate((*it)->pareto_frontier);
			}
		}
	}
	else
	{
		for (int l = 1; l < bdd->num_layers; ++l)
		{
			// cout << "\tLayer " << l << " - size = " << bdd->layers[l].size() << '\n';
			total_sol_per_layer = 0;

			// iterate on layers
			for (vector<Node *>::iterator it = bdd->layers[l].begin(); it != bdd->layers[l].end(); ++it)
			{

				Node *node = (*it);
				int id = node->index;

				// Request frontier
				node->pareto_frontier = mgmr->request();

				// add incoming zero arcs
				for (vector<Node *>::iterator prev = (*it)->prev[0].begin();
					 prev != (*it)->prev[0].end(); ++prev)
				{
					stats->num_comparisons += node->pareto_frontier->merge(*((*prev)->pareto_frontier),
																		   0,
																		   (*prev)->weights[0]);
				}

				// add incoming one arcs
				for (vector<Node *>::iterator prev = (*it)->prev[1].begin();
					 prev != (*it)->prev[1].end(); ++prev)
				{
					stats->num_comparisons += node->pareto_frontier->merge(*((*prev)->pareto_frontier),
																		   1,
																		   (*prev)->weights[1]);
				}

				// TODO
				total_sol_per_layer += (node->pareto_frontier->sols.size());
			}
			// cout << l << ": " << total_sol_per_layer << " " << bdd->layers[l].size() << " " << total_sol_per_layer / bdd->layers[l].size() << endl;

			if (dominance_strategy > 0)
			{
				init = clock();
				BDDMultiObj::filter_dominance(bdd, l, problem_type, dominance_strategy, stats);
				stats->pareto_dominance_time += clock() - init;
			}

			// Deallocate frontier from previous layer
			for (vector<Node *>::iterator it = bdd->layers[l - 1].begin(); it != bdd->layers[l - 1].end(); ++it)
			{
				mgmr->deallocate((*it)->pareto_frontier);
			}
		}
	}

	// cout << "Filtering time: " << (double)time_filter/CLOCKS_PER_SEC << endl;

	// Erase memory
	delete mgmr;
	return bdd->get_terminal()->pareto_frontier;
}

//
// Expand pareto frontier / topdown version
//
inline void expand_layer_topdown(BDD *bdd, const int l, const bool maximization, ParetoFrontierManager *mgmr)
{
	Node *node = NULL;
	if (maximization)
	{
		for (int i = 0; i < bdd->layers[l].size(); ++i)
		{
			node = bdd->layers[l][i];
			// Request frontier
			node->pareto_frontier = mgmr->request();

			// add incoming one arcs
			for (vector<Node *>::iterator prev = node->prev[1].begin(); prev != node->prev[1].end(); ++prev)
			{
				node->pareto_frontier->merge(*((*prev)->pareto_frontier), 1, (*prev)->weights[1]);
			}

			// add incoming zero arcs
			for (vector<Node *>::iterator prev = node->prev[0].begin(); prev != node->prev[0].end(); ++prev)
			{
				node->pareto_frontier->merge(*((*prev)->pareto_frontier), 0, (*prev)->weights[0]);
			}
		}
	}
	else
	{
		for (int i = 0; i < bdd->layers[l].size(); ++i)
		{
			node = bdd->layers[l][i];
			// Request frontier
			node->pareto_frontier = mgmr->request();

			// add incoming zero arcs
			for (vector<Node *>::iterator prev = node->prev[0].begin(); prev != node->prev[0].end(); ++prev)
			{
				node->pareto_frontier->merge(*((*prev)->pareto_frontier), 0, (*prev)->weights[0]);
			}

			// add incoming one arcs
			for (vector<Node *>::iterator prev = node->prev[1].begin(); prev != node->prev[1].end(); ++prev)
			{
				node->pareto_frontier->merge(*((*prev)->pareto_frontier), 1, (*prev)->weights[1]);
			}
		}
	}

	// // BDDMultiObj::filter_dominance_knapsack(bdd, l);
	// BDDMultiObj::filter_completion(bdd, l);

	// deallocate previous layer
	for (int i = 0; i < bdd->layers[l - 1].size(); ++i)
	{
		mgmr->deallocate(bdd->layers[l - 1][i]->pareto_frontier);
	}
}

//
// Expand pareto frontier / bottomup version
//
inline void expand_layer_bottomup(BDD *bdd, const int l, const bool maximization, ParetoFrontierManager *mgmr)
{
	Node *node;
	if (maximization)
	{
		for (int i = 0; i < bdd->layers[l].size(); ++i)
		{
			node = bdd->layers[l][i];

			// Request frontier
			node->pareto_frontier_bu = mgmr->request();

			// add outgoing one arcs
			if (node->arcs[1] != NULL)
			{
				node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu), 1, node->weights[1]);
			}

			// add outgoing zero arcs
			if (node->arcs[0] != NULL)
			{
				node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu), 0, node->weights[0]);
			}
		}
	}
	else
	{
		for (int i = 0; i < bdd->layers[l].size(); ++i)
		{
			node = bdd->layers[l][i];

			// Request frontier
			node->pareto_frontier_bu = mgmr->request();

			// add outgoing zero arcs
			if (node->arcs[0] != NULL)
			{
				node->pareto_frontier_bu->merge(*(node->arcs[0]->pareto_frontier_bu), 0, node->weights[0]);
			}

			// add outgoing one arcs
			if (node->arcs[1] != NULL)
			{
				node->pareto_frontier_bu->merge(*(node->arcs[1]->pareto_frontier_bu), 1, node->weights[1]);
			}
		}
	}
	// deallocate next layer
	for (int i = 0; i < bdd->layers[l + 1].size(); ++i)
	{
		mgmr->deallocate(bdd->layers[l + 1][i]->pareto_frontier_bu);
	}
}

//
// Topdown value of a node (for dynamic layer selection)
//
inline int topdown_layer_value(BDD *bdd, Node *node)
{
	int total = 0;
	for (int t = 0; t < 2; ++t)
	{
		if (node->arcs[t] != NULL)
		{
			total += node->pareto_frontier->get_num_sols();
		}
	}
	return total;
}

//
// Bottomup value of a node (for dynamic layer selection)
//
inline int bottomup_layer_value(BDD *bdd, Node *node)
{
	int total = 0;
	for (int t = 0; t < 2; ++t)
	{
		total += node->pareto_frontier_bu->get_num_sols() * node->prev[t].size();
	}
	return 1.5 * total;
}

// Comparator for node selection in convolution

struct CompareNode
{
	bool operator()(const Node *nodeA, const Node *nodeB)
	{
		return (nodeA->pareto_frontier->get_sum() + nodeA->pareto_frontier_bu->get_sum()) > (nodeB->pareto_frontier->get_sum() + nodeB->pareto_frontier_bu->get_sum());
	}
};

//
// Find pareto frontier using dynamic layer cutset
//
ParetoFrontier *BDDMultiObj::pareto_frontier_dynamic_layer_cutset(BDD *bdd, bool maximization, const int problem_type, const int dominance_strategy, MultiObjectiveStats *stats)
{
	// Initialize stats
	stats->pareto_dominance_time = 0;
	stats->pareto_dominance_filtered = 0;
	clock_t time_filter = 0, init;

	// Create pareto frontier manager
	ParetoFrontierManager *mgmr = new ParetoFrontierManager(bdd->get_width());

	// Create root and terminal frontiers
	// ObjType sol[NOBJS];
	// memset(sol, 0, sizeof(ObjType) * NOBJS);

	vector<int> x_root;
	vector<int> obj_root(NOBJS, 0);
	Solution rootSol(x_root, obj_root);
	bdd->get_root()->pareto_frontier = mgmr->request();
	bdd->get_root()->pareto_frontier->add(rootSol);

	vector<int> x_term;
	vector<int> obj_term(NOBJS, 0);
	Solution termSol(x_term, obj_term);
	bdd->get_terminal()->pareto_frontier_bu = mgmr->request();
	bdd->get_terminal()->pareto_frontier_bu->add(termSol);

	// Current layers
	int layer_topdown = 0;
	int layer_bottomup = bdd->num_layers - 1;

	// Value of layer
	int val_topdown = 0;
	int val_bottomup = 0;

	int old_topdown = -1;
	while (layer_topdown != layer_bottomup)
	{
		//		if (layer_topdown <= 3) {
		if (val_topdown <= val_bottomup)
		{
			// Expand topdown
			expand_layer_topdown(bdd, ++layer_topdown, maximization, mgmr);
			// Recompute layer value
			val_topdown = 0;
			for (int i = 0; i < bdd->layers[layer_topdown].size(); ++i)
			{
				val_topdown += topdown_layer_value(bdd, bdd->layers[layer_topdown][i]);
			}
			if (dominance_strategy > 0)
			{
				init = clock();
				BDDMultiObj::filter_dominance(bdd, layer_topdown, problem_type, dominance_strategy, stats);
				stats->pareto_dominance_time += clock() - init;
			}
		}
		else
		{
			// Expand layer bottomup
			expand_layer_bottomup(bdd, --layer_bottomup, maximization, mgmr);
			// Recompute layer value
			val_bottomup = 0;
			for (int i = 0; i < bdd->layers[layer_bottomup].size(); ++i)
			{
				val_bottomup += bottomup_layer_value(bdd, bdd->layers[layer_bottomup][i]);
			}
		}

		// if (layer_topdown != old_topdown && (layer_bottomup - layer_topdown <= 3)) {
		// 	cout << "\nFiltering..." << endl;
		// 	old_topdown = layer_topdown;
		// 	BDDMultiObj::filter_dominance_knapsack(bdd, layer_topdown);
		// }

		// cout << "\tTD=" << layer_topdown << "\tBU=" << layer_bottomup << "\tV-TD=" << val_topdown << "\tV-BU=" << val_bottomup << endl;
	}

	// Save stats
	stats->layer_coupling = layer_topdown;

	// Coupling
	// cout << "\nCoupling..." << endl;

	vector<Node *> &cutset = bdd->layers[layer_topdown];
	// cout << "\tCutset size: " << cutset.size() << endl;

	// cout << "\tsorting..." << endl;
	sort(cutset.begin(), cutset.end(), CompareNode());

	// Compute expected frontier size
	long int expected_size = 0;
	for (int i = 0; i < cutset.size(); ++i)
	{
		expected_size += cutset[i]->pareto_frontier->get_num_sols() * cutset[i]->pareto_frontier_bu->get_num_sols();
	}
	expected_size = 10000;

	ParetoFrontier *paretoFrontier = new ParetoFrontier;
	// paretoFrontier->sols.reserve(expected_size * NOBJS);

	// cout << "\tconvoluting..." << endl;
	for (int i = 0; i < cutset.size(); ++i)
	{
		Node *node = cutset[i];
		assert(node->pareto_frontier != NULL);
		assert(node->pareto_frontier_bu != NULL);
		// cout << "\t\tNode " << node->layer << "," << node->index << endl;
		paretoFrontier->convolute(*(node->pareto_frontier), *(node->pareto_frontier_bu));
	}

	// cout << "\tdeallocating..." << endl;
	// cout << endl << "Filtering time: " << (double)time_filter/CLOCKS_PER_SEC << endl;

	// deallocate manager
	delete mgmr;

	// return pareto frontier
	return paretoFrontier;
}

//
// Filter layer based on dominance
//
void BDDMultiObj::filter_dominance(BDD *bdd,
								   const int layer,
								   const int problem_type,
								   const int dominance_strategy,
								   MultiObjectiveStats *stats)
{
	if (problem_type == 1 && dominance_strategy > 0)
	{
		filter_dominance_knapsack(bdd, layer, stats);
	}
}

//
// Filter layer based on dominance / knapsack
//
void BDDMultiObj::filter_dominance_knapsack(BDD *bdd, const int layer, MultiObjectiveStats *stats)
{
	if (bdd->layers[layer].size() <= 1)
	{
		return;
	}

	vector<intpair> node_order_weight;
	node_order_weight.reserve(bdd->layers[layer].size());
	for (int i = 0; i < bdd->layers[layer].size(); ++i)
	{
		if (!bdd->layers[layer][i]->pareto_frontier->sols.empty())
		{
			node_order_weight.push_back(intpair(i, bdd->layers[layer][i]->min_weight));
		}
	}

	std::sort(node_order_weight.begin(), node_order_weight.end(), IntPairLargestToSmallestComp);
	const int order_size = static_cast<int>(node_order_weight.size());
	if (order_size <= 1)
	{
		return;
	}

	for (int i = 0; i < order_size - 1; ++i)
	{
		Node *node1 = bdd->layers[layer][node_order_weight[i].first];
		int num_dominated = 0;

		for (int j = i + 1; j < std::min(order_size, i + 3); ++j)
		{
			Node *node2 = bdd->layers[layer][node_order_weight[j].first];

			// Parent-collision false-positive guard inherited from network/common.
			if (node1->prev[0].size() + node1->prev[1].size() == 1 &&
				node2->prev[0].size() + node2->prev[1].size() == 1)
			{
				if (node1->prev[0].size() > 0 && node2->prev[1].size() > 0 &&
					node1->prev[0][0] == node2->prev[1][0])
				{
					continue;
				}
				if (node1->prev[1].size() > 0 && node2->prev[0].size() > 0 &&
					node1->prev[1][0] == node2->prev[0][0])
				{
					continue;
				}
			}

			for (SolutionList::iterator it1 = node1->pareto_frontier->sols.begin();
				 it1 != node1->pareto_frontier->sols.end();)
			{
				bool dominated = false;
				for (SolutionList::iterator it2 = node2->pareto_frontier->sols.begin();
					 it2 != node2->pareto_frontier->sols.end(); ++it2)
				{
					dominated = true;
					for (int p = 0; p < NOBJS && dominated; ++p)
					{
						dominated = (it2->obj[p] >= it1->obj[p]);
					}
					if (dominated)
					{
						break;
					}
				}

				if (dominated)
				{
					it1 = node1->pareto_frontier->sols.erase(it1);
					++num_dominated;
				}
				else
				{
					++it1;
				}
			}

			if (node1->pareto_frontier->sols.empty())
			{
				break;
			}
		}

		if (num_dominated > 0)
		{
			assert(stats != NULL);
			stats->pareto_dominance_filtered += num_dominated;
		}
	}
}
