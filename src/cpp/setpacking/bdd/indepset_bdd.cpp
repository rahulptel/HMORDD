// ----------------------------------------------------------
// Indepset BDD Constructor - Implementation
// ----------------------------------------------------------

#include <deque>

#include "indepset_bdd.hpp"
#include "../../common/bdd/bdd_alg.hpp"

#include "../../common/util/util.hpp"

using namespace boost;

//
// Generate next layer in BDD
//
bool IndepSetBDDConstructor::generate_next_layer()
{

	// cout << "\nCreating IndepSet BDD..." << endl;
	if (l < inst->graph->n_vertices + 1)
	{
		states[next].clear();
		// select next vertex
		vertex = var_layer[l - 1];

		// set weights for one arc
		one_weights = new ObjType[NOBJS];
		for (int p = 0; p < NOBJS; ++p)
		{
			one_weights[p] = objs[p][vertex];
		}

		// cout << "\tLayer " << l << " - vertex=" << vertex << " - size=" << states[iter].size() << '\n';

		BOOST_FOREACH (StateNodeMap::value_type i, states[iter])
		{
			State state = *(i.first);
			Node *node = i.second;
			bool was_set = state[vertex];

			// zero arc
			state.set(vertex, false);
			it = states[next].find(&state);
			if (it == states[next].end())
			{
				Node *new_node = bdd->add_node(l);
				// State *new_state = alloc.request();
				// (*new_state) = state;
				// states[next][new_state] = new_node;
				new_node->setpack_state = state;
				states[next][&new_node->setpack_state] = new_node;

				node->add_out_arc(new_node, 0);
				node->set_arc_weights(0, zero_weights);
			}
			else
			{
				node->add_out_arc(it->second, 0);
				node->set_arc_weights(0, zero_weights);
			}

			// one arc
			if (was_set)
			{
				state &= inst->adj_mask_compl[vertex];
				it = states[next].find(&state);
				if (it == states[next].end())
				{
					Node *new_node = bdd->add_node(l);
					// State *new_state = alloc.request();
					// (*new_state) = state;
					new_node->setpack_state = state;
					states[next][&new_node->setpack_state] = new_node;

					node->add_out_arc(new_node, 1);
					node->set_arc_weights(1, one_weights);
				}
				else
				{
					node->add_out_arc(it->second, 1);
					node->set_arc_weights(1, one_weights);
				}
			}

			// deallocate node state
			// alloc.deallocate(i.first);
		}

		// invert iter and next
		next = !next;
		iter = !iter;

		++l;
		if (l < inst->graph->n_vertices + 1)
		{
			return false;
		}
	}
	return true;
}

//
// Create BDD
//
void IndepSetBDDConstructor::generate_exact()
{
	// cout << "\nCreating IndepSet BDD..." << endl;
	l = 1;
	bool is_done;
	do
	{
		set_var_layer(-1);
		is_done = generate_next_layer();
	} while (!is_done);
}

bool CardinalityDescendingComparator(Node *a, Node *b)
{
	return a->setpack_state.count() > b->setpack_state.count();
}

bool CardinalityAscendingComparator(Node *a, Node *b)
{
	return a->setpack_state.count() < b->setpack_state.count();
}

void IndepSetBDDConstructor::restrict_layer(int rest_width, int node_select)
{
	// Get the current layer
	auto &layer = bdd->layers[l - 1];

	// Define a comparator based on node_select criterion
	switch (node_select)
	{
	case 1:
		std::sort(layer.begin(), layer.end(), CardinalityDescendingComparator);
		break;
	case 2:
		std::sort(layer.begin(), layer.end(), CardinalityAscendingComparator);
		break;
	default:
		std::sort(layer.begin(), layer.end(), CardinalityDescendingComparator);
		break;
	}

	// Remove nodes beyond rest_width
	while (layer.size() > rest_width)
	{
		Node *node_to_delete = layer.back();
		layer.pop_back();

		// Remove the node using the BDD utility function
		bdd->remove_node(node_to_delete);
	}
}

void IndepSetBDDConstructor::generate_restricted_dd(int rest_width, int node_select)
{

	l = 1;
	bool is_done;
	do
	{
		set_var_layer(-1);
		is_done = generate_next_layer();
		// Restrict
		if (bdd->layers[l - 1].size() > rest_width)
		{
			restrict_layer(rest_width, node_select);
			fix_state_map();
		}

	} while (!is_done);
}

void IndepSetBDDConstructor::fix_state_map()
{
	// If the last layer is approximated update the states[iter]
	if (states[iter].size() > bdd->layers[l - 1].size())
	{
		states[iter].clear();
		for (int k = 0; k < bdd->layers[l - 1].size(); ++k)
		{
			states[iter][&bdd->layers[l - 1][k]->setpack_state] = bdd->layers[l - 1][k];
		}
	}
}

void IndepSetBDDConstructor::set_var_layer(int v)
{
	// Use dynamically provided vertex to build next layer
	if (v > -1)
	{
		var_layer[l - 1] = v;
	}
	// Use statically provided vertex during reset
	else if (order_provided)
	{
		var_layer[l - 1] = l - 1;
	}
	// Select vertex dynamically based on min-state heuristic
	else
	{
		var_layer[l - 1] = choose_next_vertex_min_size_next_layer(states[iter]);
	}
}