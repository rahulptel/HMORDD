/*
 * --------------------------------------------------------
 * Independent set data structure
 * --------------------------------------------------------
 */

#ifndef INSTANCE_HPP_
#define INSTANCE_HPP_

#include <cstring>
#include <fstream>
#include <iostream>
#include <boost/dynamic_bitset.hpp>

using namespace std;


//
// Graph structure
//
struct Graph {

	int                         n_vertices;         /**< |V| */
	int                         n_edges;            /**< |E| */
	double*		      weights;		  /**< weight of each vertex */

	bool**                      adj_m;              /**< adjacent matrix */
	vector< vector<int> >       adj_list;           /**< adjacent list */


	/** Set two vertices as adjacents */
	void set_adj(int i, int j);

	/** Check if two vertices are adjancent */
	bool is_adj(int i, int j);

	/** Empty constructor */
	Graph();

	/** Create an isomorphic graph according to a vertex mapping */
	Graph(Graph* graph, vector<int>& mapping);

	/** Constructor with number of vertices */
	Graph(int num_vertices);

	/** Add edge */
	void add_edge(int i, int j);

	/** Remove edge */
	void remove_edge(int i, int j);

	/** Return degree of a vertex */
	int degree( int v ) { return adj_list[v].size(); }

	/** Print graph */
	void print();
};



//
// Independent set instance structure
//
struct IndepSetInst {

	Graph*              				graph;             	// independent set graph
	vector< boost::dynamic_bitset<> >	adj_mask_compl;	 	// complement mask of adjacencies
	vector<vector<int>> obj_coeffs;

	/** Create empty instance */
	IndepSetInst() { }

	/** Create from graph */
	IndepSetInst(Graph* _graph);

	IndepSetInst(int n_vertices, vector<vector<int>> edges, vector<vector<int>> obj_ceoffs);
};



/*
 * -----------------------------------------------------
 * Inline implementations: Graph
 * -----------------------------------------------------
 */


/**
 * Empty constructor
 */
inline Graph::Graph() : n_vertices(0), n_edges(0), weights(NULL), adj_m(NULL) {
}


/**
 * Constructor with number of vertices
 **/
inline Graph::Graph(int num_vertices)
: n_vertices(num_vertices), n_edges(0), weights(NULL)
{
	adj_m = new bool*[ num_vertices ];
	for (int i = 0; i < num_vertices; ++i) {
		adj_m[i] = new bool[ num_vertices ];
		memset( adj_m[i], false, sizeof(bool) * num_vertices );
	}
	adj_list.resize(num_vertices);
}


/**
 * Check if two vertices are adjacent
 */
inline bool Graph::is_adj(int i, int j) {
	assert(i >= 0);
	assert(j >= 0);
	assert(i < n_vertices);
	assert(j < n_vertices);
	return adj_m[i][j];
}


/**
 * Set two vertices as adjacent
 */
inline void Graph::set_adj(int i, int j) {
	assert(i >= 0);
	assert(j >= 0);
	assert(i < n_vertices);
	assert(j < n_vertices);

	// check if already adjacent
	if (adj_m[i][j])
		return;

	// add to adjacent matrix and list
	adj_m[i][j] = true;
	adj_list[i].push_back(j);
}



/**
 * Add edge
 **/
inline void Graph::add_edge(int i, int j) {
	assert(i >= 0);
	assert(j >= 0);
	assert(i < n_vertices);
	assert(j < n_vertices);

	// check if already adjacent
	if (adj_m[i][j])
		return;

	// add to adjacent matrix and list
	adj_m[i][j] = true;
	adj_m[j][i] = true;
	adj_list[i].push_back(j);
	adj_list[j].push_back(i);

	n_edges++;
}

/**
 * Remove edge
 **/
inline void Graph::remove_edge(int i, int j) {
	assert(i >= 0);
	assert(j >= 0);
	assert(i < n_vertices);
	assert(j < n_vertices);

	// check if already adjacent
	if (!adj_m[i][j])
		return;

	// add to adjacent matrix and list
	adj_m[i][j] = false;
	adj_m[j][i] = false;

	for (int v = 0; v < (int)adj_list[i].size(); ++v) {
		if ( adj_list[i][v] == j ) {
			adj_list[i][v] = adj_list[i].back();
			adj_list[i].pop_back();
			break;
		}
	}

	for (int v = 0; v < (int)adj_list[j].size(); ++v) {
		if ( adj_list[j][v] == i ) {
			adj_list[j][v] = adj_list[j].back();
			adj_list[j].pop_back();
			break;
		}
	}
	n_edges--;
}


/*
 * -----------------------------------------------------
 * Inline implementations: Independent Set
 * -----------------------------------------------------
 */




/** Create IndepsetInst from graph */
inline IndepSetInst::IndepSetInst(Graph* _graph) : graph(_graph) {
	// create complement mask of adjacencies
	adj_mask_compl.resize(graph->n_vertices);
	for( int v = 0; v < graph->n_vertices; v++ ) {

		adj_mask_compl[v].resize(graph->n_vertices, true);
		for( int w = 0; w < graph->n_vertices; w++ ) {
			if( graph->is_adj(v,w) ) {
				adj_mask_compl[v].set(w, false);
			}
		}

		// we assume here a vertex is adjacent to itself
		adj_mask_compl[v].set(v, false);
	}
}


inline IndepSetInst::IndepSetInst(int n_vars, vector<vector<int>> edges, vector<vector<int>> _obj_coeffs)
{
	graph = new Graph(n_vars);

	const int edge_count = static_cast<int>(edges.size());
	for (int i = 0; i < edge_count; ++i)
	{
		graph->add_edge(edges[i][0], edges[i][1]);
	}
	// graph->print();
	// cout << "\tAuxiliary graph for set packing:" << endl;
	// cout << "\t\tnumber of vertices: " << graph->n_vertices << endl;
	// cout << "\t\tnumber of edges: " << graph->n_edges << endl;

	// create complement mask of adjacencies
	adj_mask_compl.resize(graph->n_vertices);
	for (int v = 0; v < graph->n_vertices; ++v)
	{
		adj_mask_compl[v].resize(graph->n_vertices, true);
		for (int w = 0; w < graph->n_vertices; w++)
		{
			if (graph->is_adj(v, w))
			{
				adj_mask_compl[v].set(w, false);
			}
		}

		// we assume here a vertex is adjacent to itself
		adj_mask_compl[v].set(v, false);
	}

	obj_coeffs = _obj_coeffs;
}




#endif /* THISANCE_HPP_ */


