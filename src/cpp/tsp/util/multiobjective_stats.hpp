#pragma once

#include <ctime>

struct MultiObjectiveStats
{
    clock_t pareto_dominance_time;
    int pareto_dominance_filtered;
    int layer_coupling;

    MultiObjectiveStats()
        : pareto_dominance_time(0), pareto_dominance_filtered(0), layer_coupling(0)
    {
    }
};
