#pragma once

#define DOMINATED -9999999

#include "../../common/util/solution.hpp"
#include "../../common/util/stats.hpp"
#include "../../common/util/util.hpp"

//
// Pareto Frontier struct
//
class ParetoFrontier
{
public:
    explicit ParetoFrontier(bool track_x = true)
        : track_x(track_x) {}

    // Tracked solutions (x,z)
    SolutionList sols;
    // Flat objective-only solutions packed by NOBJS
    vector<ObjType> sols_flat;
    bool track_x;

    bool uses_flat() const
    {
        return !track_x;
    }

    bool empty() const
    {
        return uses_flat() ? sols_flat.empty() : sols.empty();
    }

    // Add element to set
    void add(Solution &new_sol);
    void add_flat(const ObjType *elem);

    // Merge pareto frontier solutions with shift
    void merge(ParetoFrontier &frontier, const ObjType *shift, int last_city);
    void merge_flat(const ParetoFrontier &frontier, const ObjType *shift);

    // Merge pareto frontier solutions with objective offset
    void merge(ParetoFrontier &frontier, Solution &offset_sol, bool offset_from_bu);
    void merge_flat_offset(const ParetoFrontier &frontier, const ObjType *offset_obj);

    // Convolute two nodes from this set to this one
    void convolute(ParetoFrontier &fA, ParetoFrontier &fB);
    void convolute_flat(const ParetoFrontier &fA, const ParetoFrontier &fB);

    // Get number of solutions
    int get_num_sols() const
    {
        return uses_flat() ? static_cast<int>(sols_flat.size() / NOBJS) : static_cast<int>(sols.size());
    }

    // Clear pareto frontier
    void clear()
    {
        sols.clear();
        sols_flat.clear();
    }

    // Print elements in set
    void print_frontier();

    // Obtain sum of points
    ObjType get_sum();
    ObjType get_sum_flat() const;

    map<string, vector<vector<int>>> get_frontier();

private:
    ObjType aux[NOBJS];
    ObjType auxB[NOBJS];

    void remove_empty_flat();
};

//
// Pareto frontier manager
//
class ParetoFrontierManager
{
public:
    // Constructor
    ParetoFrontierManager()
        : track_x(true) {}

    ParetoFrontierManager(int size, bool track_x = true)
        : track_x(track_x)
    {
        frontiers.reserve(size);
    }

    // Destructor
    ~ParetoFrontierManager()
    {
        for (int i = 0; i < static_cast<int>(frontiers.size()); ++i)
        {
            delete frontiers[i];
        }
    }

    // Request pareto frontier
    ParetoFrontier *request()
    {
        if (frontiers.empty())
        {
            return new ParetoFrontier(track_x);
        }
        ParetoFrontier *f = frontiers.back();
        f->clear();
        f->track_x = track_x;
        frontiers.pop_back();
        return f;
    }

    // Return frontier to allocation
    void deallocate(ParetoFrontier *frontier)
    {
        frontiers.push_back(frontier);
    }

    // Preallocated array set
    vector<ParetoFrontier *> frontiers;
    bool track_x;
};
