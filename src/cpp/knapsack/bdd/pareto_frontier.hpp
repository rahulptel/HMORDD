// ----------------------------------------------------------
// Pareto Frontier classes
// ----------------------------------------------------------

#ifndef PARETO_FRONTIER_HPP_
#define PARETO_FRONTIER_HPP_

#define DOMINATED -9999999
#define EPS 0.0001

#include <algorithm>
#include <iostream>
#include <fstream>
#include <limits>
#include <vector>
#include <list>
#include <iterator>
#include <map>
#include "../../common/util/util.hpp"
#include "../../common/util/solution.hpp"

using namespace std;

//
// Pareto Frontier struct
//
class ParetoFrontier
{
public:
    explicit ParetoFrontier(bool track_x = true, bool count_comparisons = true)
        : track_x(track_x), count_comparisons(count_comparisons) {}

    // Tracked solutions (x,z)
    SolutionList sols;
    // Flat objective-only solutions (packed blocks of size NOBJS)
    vector<ObjType> sols_flat;
    bool track_x;
    bool count_comparisons;

    bool uses_flat() const
    {
        return !track_x;
    }

    bool empty() const
    {
        return uses_flat() ? sols_flat.empty() : sols.empty();
    }

    // Add element to set (tracked)
    void add(Solution &sol);
    // Add element to set (flat)
    void add_flat(const ObjType *elem);

    // Merge pareto frontier solutions with shift
    size_t merge(ParetoFrontier &frontier, int arc_type, ObjType *shift);
    size_t merge_flat(const ParetoFrontier &frontier, const ObjType *shift);

    // Merge pareto frontier solutions with offset (used in coupling)
    void merge(ParetoFrontier &frontier, Solution &offset_sol, bool offset_from_bu);
    void merge_flat_offset(const ParetoFrontier &frontier, const ObjType *offset_obj);

    // Convolute two nodes from this set to this one
    void convolute(ParetoFrontier &fA, ParetoFrontier &fB);
    void convolute_flat(const ParetoFrontier &fA, const ParetoFrontier &fB);

    // Get number of solutions
    int get_num_sols()
    {
        return uses_flat() ? static_cast<int>(sols_flat.size() / NOBJS) : static_cast<int>(sols.size());
    }

    // Remove pre-set dominated solutions (flat mode only)
    void remove_dominated()
    {
        if (uses_flat())
        {
            remove_empty_flat();
        }
    }

    // Clear pareto frontier
    void clear()
    {
        sols.clear();
        sols_flat.clear();
    }

    // Print elements in set
    void print();

    // Check consistency
    // bool check_consistency();

    // Obtain sum of points
    ObjType get_sum();
    ObjType get_sum_flat() const;

    map<string, vector<vector<int>>> get_frontier();

private:
    // Auxiliaries
    ObjType aux[NOBJS];
    ObjType auxB[NOBJS];
    vector<ObjType *> elems;
    // vector<int> aux;
    // ObjType *aux;

    // Remove empty elements in flat objective array
    void remove_empty_flat();
};

//
// Pareto frontier manager
//
class ParetoFrontierManager
{
public:
    // Constructor
    ParetoFrontierManager() : track_x(true), count_comparisons(true) {}
    ParetoFrontierManager(int size, bool track_x = true, bool count_comparisons = true)
        : track_x(track_x), count_comparisons(count_comparisons)
    {
        frontiers.reserve(size);
    }

    // Destructor
    ~ParetoFrontierManager()
    {
        for (int i = 0; i < frontiers.size(); ++i)
        {
            delete frontiers[i];
        }
    }

    // Request pareto frontier
    ParetoFrontier *request()
    {
        if (frontiers.empty())
        {
            return new ParetoFrontier(track_x, count_comparisons);
        }
        ParetoFrontier *f = frontiers.back();
        f->clear();
        f->track_x = track_x;
        f->count_comparisons = count_comparisons;
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
    bool count_comparisons;
};

// Modify
//
// Add element to set
//
inline void ParetoFrontier::add(Solution &sol)
{
    if (uses_flat())
    {
        add_flat(sol.obj.data());
        return;
    }

    bool dominates;
    bool dominated;
    for (SolutionList::iterator it = sols.begin(); it != sols.end();)
    {
        // check status of foreign solution w.r.t. current frontier solution
        dominates = true;
        dominated = true;
        for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
        {
            dominates &= (sol.obj[o] >= (*it).obj[o]);
            dominated &= (sol.obj[o] <= (*it).obj[o]);
        }
        if (dominated)
        {
            // if foreign solution is dominated, nothing needs to be done
            return;
        }
        else if (dominates)
        {
            // solution dominates iterate
            it = sols.erase(it);
        }
        else
        {
            ++it;
        }
    }
    sols.insert(sols.end(), sol);
}

inline void ParetoFrontier::add_flat(const ObjType *elem)
{
    bool must_add = true;
    bool dominates;
    bool dominated;
    for (int i = 0; i < static_cast<int>(sols_flat.size()); i += NOBJS)
    {
        dominates = true;
        dominated = true;
        for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
        {
            dominates &= (elem[o] >= sols_flat[i + o]);
            dominated &= (elem[o] <= sols_flat[i + o]);
        }
        if (dominated)
        {
            return;
        }
        else if (dominates)
        {
            if (must_add)
            {
                std::copy(elem, elem + NOBJS, sols_flat.begin() + i);
                must_add = false;
            }
            else
            {
                sols_flat[i] = DOMINATED;
            }
        }
    }

    if (must_add)
    {
        sols_flat.insert(sols_flat.end(), elem, elem + NOBJS);
    }
    remove_empty_flat();
}

//
// Merge pareto frontier into existing set considering shift
//
inline size_t ParetoFrontier::merge(ParetoFrontier &frontier, int arc_type, ObjType *shift)
{
    if (uses_flat())
    {
        return merge_flat(frontier, shift);
    }

    bool must_add;
    bool dominates;
    bool dominated;
    size_t num_comparisons = 0;

    // add artificial solution to avoid rechecking dominance between elements in the
    // set to be merged
    Solution dummy;
    SolutionList::iterator end = sols.insert(sols.end(), dummy);
    for (SolutionList::iterator itParent = frontier.sols.begin();
         itParent != frontier.sols.end();
         ++itParent)
    {
        // update auxiliary
        for (int o = 0; o < NOBJS; ++o)
        {
            aux[o] = (*itParent).obj[o] + shift[o];
        }
        must_add = true;
        // Compare the incoming aux solution with the sols on the current node
        for (SolutionList::iterator itCurr = sols.begin();
             itCurr != end;)
        {
            // check status of foreign solution w.r.t. current frontier solution
            dominates = true;
            dominated = true;
            for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
            {
                dominates &= (aux[o] >= (*itCurr).obj[o]);
                dominated &= (aux[o] <= (*itCurr).obj[o]);
                if (count_comparisons)
                {
                    num_comparisons += 1;
                }
            }
            if (dominated)
            {
                // if foreign solution is dominated, just stop loop
                must_add = false;
                break;
            }
            else if (dominates)
            {

                itCurr = sols.erase(itCurr);
            }
            else
            {
                ++itCurr;
            }
        }
        // if solution has not been added already, append element to the end
        if (must_add)
        {
            vector<int> new_x;
            if (track_x)
            {
                new_x = (*itParent).x;
                new_x.push_back(arc_type);
            }
            Solution new_sol(new_x, (*itParent).obj);
            for (int o = 0; o < NOBJS; ++o)
            {
                new_sol.obj[o] = aux[o];
            }
            // Push new solution at the end of the current solution list
            sols.push_back(new_sol);
        }
    }
    sols.erase(end);

    return num_comparisons;
}

inline size_t ParetoFrontier::merge_flat(const ParetoFrontier &frontier, const ObjType *shift)
{
    size_t num_comparisons = 0;
    int end = static_cast<int>(sols_flat.size());
    bool must_add;
    bool dominates;
    bool dominated;

    for (int j = 0; j < static_cast<int>(frontier.sols_flat.size()); j += NOBJS)
    {
        for (int o = 0; o < NOBJS; ++o)
        {
            aux[o] = frontier.sols_flat[j + o] + shift[o];
        }

        must_add = true;
        for (int i = 0; i < end; i += NOBJS)
        {
            if (sols_flat[i] == DOMINATED)
            {
                continue;
            }

            dominates = true;
            dominated = true;
            for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
            {
                dominates &= (aux[o] >= sols_flat[i + o]);
                dominated &= (aux[o] <= sols_flat[i + o]);
                if (count_comparisons)
                {
                    num_comparisons += 1;
                }
            }
            if (dominated)
            {
                must_add = false;
                break;
            }
            else if (dominates)
            {
                if (must_add)
                {
                    std::copy(aux, aux + NOBJS, sols_flat.begin() + i);
                    must_add = false;
                }
                else
                {
                    sols_flat[i] = DOMINATED;
                }
            }
        }

        if (must_add)
        {
            sols_flat.insert(sols_flat.end(), aux, aux + NOBJS);
        }
    }

    remove_empty_flat();
    return num_comparisons;
}

// //
// // Merge pareto frontier into existing set considering shift
// //
// inline void ParetoFrontier::merge_after_convolute(ParetoFrontier &frontier, Solution &sol, bool reverse_outer)
// {
//     bool must_add;
//     bool dominates;
//     bool dominated;

//     // add artificial solution to avoid rechecking dominance between elements in the
//     // set to be merged
//     Solution dummy;
//     SolutionList::iterator end = sols.insert(sols.end(), dummy);

//     for (SolutionList::iterator itParent = frontier.sols.begin();
//          itParent != frontier.sols.end();
//          ++itParent)
//     {
//         // update auxiliary
//         for (int o = 0; o < NOBJS; ++o)
//         {
//             aux[o] = itParent->obj[o] + sol.obj[o];
//         }
//         must_add = true;
//         // Compare the incoming aux solution with the sols on the current node
//         for (SolutionList::iterator itCurr = sols.begin();
//              itCurr != end;)
//         {
//             // check status of foreign solution w.r.t. current frontier solution
//             dominates = true;
//             dominated = true;
//             for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
//             {
//                 dominates &= (aux[o] >= itCurr->obj[o]);
//                 dominated &= (aux[o] <= itCurr->obj[o]);
//             }
//             if (dominated)
//             {
//                 // if foreign solution is dominated, just stop loop
//                 must_add = false;
//                 break;
//             }
//             else if (dominates)
//             {
//                 itCurr = sols.erase(itCurr);
//             }
//             else
//             {
//                 ++itCurr;
//             }
//         }
//         // if solution has not been added already, append element to the end
//         if (must_add)
//         {

//             if (reverse_outer)
//             {
//                 Solution new_solution(sol.x, aux);
//                 reverse(itParent->x.begin(), itParent->x.end());
//                 new_solution.x.insert(new_solution.x.end(), itParent->x.begin(), itParent->x.end());
//                 sols.push_back(new_solution);
//             }
//             else
//             {
//                 Solution new_solution(itParent->x, aux);
//                 reverse(sol.x.begin(), sol.x.end());
//                 new_solution.x.insert(new_solution.x.end(), sol.x.begin(), sol.x.end());
//                 sols.push_back(new_solution);
//             }
//         }
//     }
//     sols.erase(end);
// }

// Merge pareto frontier into existing set considering shift
//
inline void ParetoFrontier::merge(ParetoFrontier &frontier, Solution &offset_sol, bool offset_from_bu)
{
    if (uses_flat())
    {
        merge_flat_offset(frontier, offset_sol.obj.data());
        return;
    }

    bool must_add;
    bool dominates;
    bool dominated;

    // add artificial solution to avoid rechecking dominance between elements in the
    // set to be merged
    Solution dummy;
    SolutionList::iterator end = sols.insert(sols.end(), dummy);

    // Reverse X variable order if the offset is from the bottom-up set.
    if (track_x && offset_from_bu)
    {
        reverse(offset_sol.x.begin(), offset_sol.x.end());
    }

    for (SolutionList::iterator itParent = frontier.sols.begin();
         itParent != frontier.sols.end();
         ++itParent)
    {
        // update auxiliary
        for (int o = 0; o < NOBJS; ++o)
        {
            aux[o] = (*itParent).obj[o] + offset_sol.obj[o];
        }
        must_add = true;
        // Compare the incoming aux solution with the sols on the current node
        for (SolutionList::iterator itCurr = sols.begin();
             itCurr != end;)
        {
            // check status of foreign solution w.r.t. current frontier solution
            dominates = true;
            dominated = true;
            for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
            {
                dominates &= (aux[o] >= itCurr->obj[o]);
                dominated &= (aux[o] <= itCurr->obj[o]);
            }
            if (dominated)
            {
                // if foreign solution is dominated, just stop loop
                must_add = false;
                break;
            }
            else if (dominates)
            {
                itCurr = sols.erase(itCurr);
            }
            else
            {
                ++itCurr;
            }
        }
        // if solution has not been added already, append element to the end
        if (must_add)
        {

            if (track_x && offset_from_bu)
            {
                Solution new_sol(itParent->x, itParent->obj);
                new_sol.x.insert(new_sol.x.end(), offset_sol.x.begin(), offset_sol.x.end());
                for (int i = 0; i < NOBJS; ++i)
                {
                    new_sol.obj[i] = aux[i];
                }
                sols.push_back(new_sol);
            }
            else if (track_x)
            {
                Solution new_sol(offset_sol.x, offset_sol.obj);
                reverse(itParent->x.begin(), itParent->x.end());
                new_sol.x.insert(new_sol.x.end(), itParent->x.begin(), itParent->x.end());
                for (int i = 0; i < NOBJS; ++i)
                {
                    new_sol.obj[i] = aux[i];
                }
                sols.push_back(new_sol);
            }
            else
            {
                vector<int> new_x;
                Solution new_sol(new_x, offset_sol.obj);
                for (int i = 0; i < NOBJS; ++i)
                {
                    new_sol.obj[i] = aux[i];
                }
                sols.push_back(new_sol);
            }
        }
    }
    sols.erase(end);
}

inline void ParetoFrontier::merge_flat_offset(const ParetoFrontier &frontier, const ObjType *offset_obj)
{
    (void)merge_flat(frontier, offset_obj);
}

//
// Print elements in set//
// // Merge pareto frontier into existing set considering shift
// //
// inline void ParetoFrontier::merge_after_convolute(ParetoFrontier &frontier, Solution &sol, bool reverse_outer)
// {
//     bool must_add;
//     bool dominates;
//     bool dominated;

//     // add artificial solution to avoid rechecking dominance between elements in the
//     // set to be merged
//     Solution dummy;
//     SolutionList::iterator end = sols.insert(sols.end(), dummy);

//     for (SolutionList::iterator itParent = frontier.sols.begin();
//          itParent != frontier.sols.end();
//          ++itParent)
//     {
//         // update auxiliary
//         for (int o = 0; o < NOBJS; ++o)
//         {
//             aux[o] = itParent->obj[o] + sol.obj[o];
//         }
//         must_add = true;
//         // Compare the incoming aux solution with the sols on the current node
//         for (SolutionList::iterator itCurr = sols.begin();
//              itCurr != end;)
//         {
//             // check status of foreign solution w.r.t. current frontier solution
//             dominates = true;
//             dominated = true;
//             for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
//             {
//                 dominates &= (aux[o] >= itCurr->obj[o]);
//                 dominated &= (aux[o] <= itCurr->obj[o]);
//             }
//             if (dominated)
//             {
//                 // if foreign solution is dominated, just stop loop
//                 must_add = false;
//                 break;
//             }
//             else if (dominates)
//             {
//                 itCurr = sols.erase(itCurr);
//             }
//             else
//             {
//                 ++itCurr;
//             }
//         }
//         // if solution has not been added already, append element to the end
//         if (must_add)
//         {

//             if (reverse_outer)
//             {
//                 Solution new_solution(sol.x, aux);
//                 reverse(itParent->x.begin(), itParent->x.end());
//                 new_solution.x.insert(new_solution.x.end(), itParent->x.begin(), itParent->x.end());
//                 sols.push_back(new_solution);
//             }
//             else
//             {
//                 Solution new_solution(itParent->x, aux);
//                 reverse(sol.x.begin(), sol.x.end());
//                 new_solution.x.insert(new_solution.x.end(), sol.x.begin(), sol.x.end());
//                 sols.push_back(new_solution);
//             }
//         }
//     }
//     sols.erase(end);
// }

//
inline void ParetoFrontier::print()
{
    if (uses_flat())
    {
        for (int i = 0; i < static_cast<int>(sols_flat.size()); i += NOBJS)
        {
            cout << "(";
            for (int o = 0; o < NOBJS - 1; ++o)
            {
                cout << sols_flat[i + o] << ",";
            }
            cout << sols_flat[i + NOBJS - 1] << ")";
            cout << endl;
        }
        return;
    }

    for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
    {
        cout << "(";
        for (int o = 0; o < NOBJS - 1; ++o)
        {
            cout << it->obj[o] << ",";
        }
        cout << it->obj[NOBJS - 1] << ")";
        cout << endl;
    }
}

//
// Convolute two nodes from this set to this one
//
inline void ParetoFrontier::convolute(ParetoFrontier &fA, ParetoFrontier &fB)
{
    if (uses_flat())
    {
        convolute_flat(fA, fB);
        return;
    }

    if (fA.sols.size() < fB.sols.size())
    {
        for (SolutionList::iterator solA = fA.sols.begin(); solA != fA.sols.end(); ++solA)
        {
            // std::copy(fA.sols.begin() + j, fA.sols.begin() + j + NOBJS, auxB);
            merge(fB, (*solA), false);
        }
    }
    else
    {
        for (SolutionList::iterator solB = fB.sols.begin(); solB != fB.sols.end(); ++solB)
        {
            // std::copy(fB.sols.begin() + j, fB.sols.begin() + j + NOBJS, auxB);
            merge(fA, (*solB), true);
        }
    }
}

inline void ParetoFrontier::convolute_flat(const ParetoFrontier &fA, const ParetoFrontier &fB)
{
    if (fA.sols_flat.size() < fB.sols_flat.size())
    {
        for (int j = 0; j < static_cast<int>(fA.sols_flat.size()); j += NOBJS)
        {
            std::copy(fA.sols_flat.begin() + j, fA.sols_flat.begin() + j + NOBJS, auxB);
            merge_flat_offset(fB, auxB);
        }
    }
    else
    {
        for (int j = 0; j < static_cast<int>(fB.sols_flat.size()); j += NOBJS)
        {
            std::copy(fB.sols_flat.begin() + j, fB.sols_flat.begin() + j + NOBJS, auxB);
            merge_flat_offset(fA, auxB);
        }
    }
}

//
// Obtain sum of points
//
inline ObjType ParetoFrontier::get_sum()
{
    if (uses_flat())
    {
        return get_sum_flat();
    }

    ObjType sum = 0;
    for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
    {
        for (int o = 0; o < NOBJS; ++o)
        {
            sum += it->obj[o];
        }
    }
    return sum;
}

inline ObjType ParetoFrontier::get_sum_flat() const
{
    ObjType sum = 0;
    for (int i = 0; i < static_cast<int>(sols_flat.size()); ++i)
    {
        sum += sols_flat[i];
    }
    return sum;
}

inline void ParetoFrontier::remove_empty_flat()
{
    if (sols_flat.empty())
    {
        return;
    }

    int last = static_cast<int>(sols_flat.size()) - NOBJS;
    while (last >= 0 && sols_flat[last] == DOMINATED)
    {
        last -= NOBJS;
    }
    if (last < 0)
    {
        sols_flat.clear();
        return;
    }

    for (int i = 0; i < last; i += NOBJS)
    {
        if (sols_flat[i] == DOMINATED)
        {
            std::copy(sols_flat.begin() + last, sols_flat.begin() + last + NOBJS, sols_flat.begin() + i);
            last -= NOBJS;
            while (last >= 0 && sols_flat[last] == DOMINATED)
            {
                last -= NOBJS;
            }
            if (last < 0)
            {
                break;
            }
        }
    }

    if (last >= 0)
    {
        sols_flat.resize(last + NOBJS);
    }
    else
    {
        sols_flat.clear();
    }
}

inline map<string, vector<vector<int>>> ParetoFrontier::get_frontier()
{
    vector<vector<int>> x_sols;
    vector<vector<int>> z_sols;

    if (uses_flat())
    {
        z_sols.reserve(get_num_sols());
        for (int i = 0; i < static_cast<int>(sols_flat.size()); i += NOBJS)
        {
            vector<int> z(NOBJS, 0);
            for (int o = 0; o < NOBJS; ++o)
            {
                z[o] = sols_flat[i + o];
            }
            z_sols.push_back(std::move(z));
        }
    }
    else
    {
        z_sols.reserve(sols.size());
        x_sols.reserve(sols.size());
        for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
        {
            x_sols.push_back((*it).x);
            z_sols.push_back((*it).obj);
        }
    }

    map<string, vector<vector<int>>> frontier;
    frontier.insert({"x", x_sols});
    frontier.insert({"z", z_sols});

    return frontier;
}

#endif
