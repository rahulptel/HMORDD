#include "frontier.hpp"
#include <algorithm>

//
// Print elements in set
//
void ParetoFrontier::print_frontier()
{
    if (uses_flat())
    {
        for (int i = 0; i < static_cast<int>(sols_flat.size()); i += NOBJS)
        {
            cout << "(";
            for (int o = 0; o < NOBJS; ++o)
            {
                cout << sols_flat[i + o];
                if (o < NOBJS - 1)
                {
                    cout << ", ";
                }
            }
            cout << ")" << endl;
        }
        return;
    }

    for (SolutionList::iterator it = sols.begin(); it != sols.end(); ++it)
    {
        (*it).print_x();
        (*it).print_obj();
    }
}

//
// Add element to set
//
void ParetoFrontier::add(Solution &new_sol)
{
    if (uses_flat())
    {
        add_flat(new_sol.obj.data());
        return;
    }

    bool dominates;
    bool dominated;
    for (SolutionList::iterator it = sols.begin(); it != sols.end();)
    {
        dominates = true;
        dominated = true;
        for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
        {
            dominates &= (new_sol.obj[o] >= (*it).obj[o]);
            dominated &= (new_sol.obj[o] <= (*it).obj[o]);
        }
        if (dominated)
        {
            return;
        }
        else if (dominates)
        {
            it = sols.erase(it);
        }
        else
        {
            ++it;
        }
    }
    sols.insert(sols.end(), new_sol);
}

void ParetoFrontier::add_flat(const ObjType *elem)
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
void ParetoFrontier::merge(ParetoFrontier &frontier, const ObjType *shift, int last_city)
{
    if (uses_flat())
    {
        merge_flat(frontier, shift);
        return;
    }

    bool must_add;
    bool dominates;
    bool dominated;
    Solution dummy;
    SolutionList::iterator end = sols.insert(sols.end(), dummy);

    for (SolutionList::iterator itParent = frontier.sols.begin();
         itParent != frontier.sols.end();
         ++itParent)
    {
        for (int o = 0; o < NOBJS; ++o)
        {
            aux[o] = (*itParent).obj[o] + shift[o];
        }

        must_add = true;
        for (SolutionList::iterator itCurr = sols.begin();
             itCurr != end;)
        {
            dominates = true;
            dominated = true;
            for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
            {
                dominates &= (aux[o] >= (*itCurr).obj[o]);
                dominated &= (aux[o] <= (*itCurr).obj[o]);
            }
            if (dominated)
            {
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

        if (must_add)
        {
            Solution new_sol((*itParent).x, (*itParent).obj);
            new_sol.x.push_back(last_city);
            for (int o = 0; o < NOBJS; ++o)
            {
                new_sol.obj[o] = aux[o];
            }
            sols.push_back(new_sol);
        }
    }
    sols.erase(end);
}

void ParetoFrontier::merge_flat(const ParetoFrontier &frontier, const ObjType *shift)
{
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
}

//
// Merge pareto frontier into existing set considering objective offset
//
void ParetoFrontier::merge(ParetoFrontier &frontier, Solution &offset_sol, bool offset_from_bu)
{
    if (uses_flat())
    {
        merge_flat_offset(frontier, offset_sol.obj.data());
        return;
    }

    bool must_add;
    bool dominates;
    bool dominated;
    Solution dummy;
    SolutionList::iterator end = sols.insert(sols.end(), dummy);

    if (offset_from_bu)
    {
        reverse(offset_sol.x.begin(), offset_sol.x.end());
    }

    for (SolutionList::iterator itParent = frontier.sols.begin();
         itParent != frontier.sols.end();
         ++itParent)
    {
        for (int o = 0; o < NOBJS; ++o)
        {
            aux[o] = (*itParent).obj[o] + offset_sol.obj[o];
        }

        must_add = true;
        for (SolutionList::iterator itCurr = sols.begin();
             itCurr != end;)
        {
            dominates = true;
            dominated = true;
            for (int o = 0; o < NOBJS && (dominates || dominated); ++o)
            {
                dominates &= (aux[o] >= (*itCurr).obj[o]);
                dominated &= (aux[o] <= (*itCurr).obj[o]);
            }
            if (dominated)
            {
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

        if (must_add)
        {
            if (offset_from_bu)
            {
                Solution new_sol((*itParent).x, (*itParent).obj);
                new_sol.x.insert(new_sol.x.end(), offset_sol.x.begin(), offset_sol.x.end());
                for (int i = 0; i < NOBJS; ++i)
                {
                    new_sol.obj[i] = aux[i];
                }
                sols.push_back(new_sol);
            }
            else
            {
                Solution new_sol(offset_sol.x, offset_sol.obj);
                reverse((*itParent).x.begin(), (*itParent).x.end());
                new_sol.x.insert(new_sol.x.end(), (*itParent).x.begin(), (*itParent).x.end());
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

void ParetoFrontier::merge_flat_offset(const ParetoFrontier &frontier, const ObjType *offset_obj)
{
    merge_flat(frontier, offset_obj);
}

//
// Convolute two nodes from this set to this one
//
void ParetoFrontier::convolute(ParetoFrontier &fA, ParetoFrontier &fB)
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
            merge(fB, (*solA), false);
        }
    }
    else
    {
        for (SolutionList::iterator solB = fB.sols.begin(); solB != fB.sols.end(); ++solB)
        {
            merge(fA, (*solB), true);
        }
    }
}

void ParetoFrontier::convolute_flat(const ParetoFrontier &fA, const ParetoFrontier &fB)
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
ObjType ParetoFrontier::get_sum()
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
            sum += (*it).obj[o];
        }
    }
    return sum;
}

ObjType ParetoFrontier::get_sum_flat() const
{
    ObjType sum = 0;
    for (int i = 0; i < static_cast<int>(sols_flat.size()); ++i)
    {
        sum += sols_flat[i];
    }
    return sum;
}

void ParetoFrontier::remove_empty_flat()
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

map<string, vector<vector<int>>> ParetoFrontier::get_frontier()
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
        x_sols.reserve(sols.size());
        z_sols.reserve(sols.size());
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
