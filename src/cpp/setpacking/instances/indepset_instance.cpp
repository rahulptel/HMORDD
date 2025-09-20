/**
 * -------------------------------------------------
 * Independent Set structure - Implementation
 * -------------------------------------------------
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "indepset_instance.hpp"
#include "../../common/util/util.hpp"

using namespace std;

//
// Create an isomorphic graph according to a vertex mapping
// Mapping description: mapping[i] = position where vertex i is in new ordering
//
//
Graph::Graph(Graph *graph, vector<int> &mapping)
	: n_vertices(graph->n_vertices), n_edges(graph->n_edges)
{
	// allocate adjacent matrix
	adj_m = new bool *[n_vertices];
	for (int i = 0; i < n_vertices; ++i)
	{
		adj_m[i] = new bool[n_vertices];
		memset(adj_m[i], false, sizeof(bool) * n_vertices);
	}

	// allocate adjacent list
	adj_list.resize(n_vertices);

	// construct graph according to mapping
	for (int i = 0; i < graph->n_vertices; ++i)
	{
		for (vector<int>::iterator it = graph->adj_list[i].begin();
			 it != graph->adj_list[i].end();
			 it++)
		{
			set_adj(mapping[i], mapping[*it]);
		}
	}
}

//
// Print graph
//
void Graph::print()
{
	cout << "Graph" << endl;
	for (int v = 0; v < n_vertices; ++v)
	{
		if (adj_list[v].size() != 0)
		{
			cout << "\t" << v << " --> ";
			for (vector<int>::iterator it = adj_list[v].begin();
				 it != adj_list[v].end();
				 ++it)
			{
				cout << *it << " ";
			}
			cout << endl;
		}
	}
	cout << endl;
}
