// engine.Automaton

#include <iostream>

#include "Automaton.h"

Automaton::Automaton(int w, int h, std::unique_ptr<Rule> r) : grid(w, h), rule(std::move(r)) {}

void Automaton::Step()
{
    Grid next(grid.getWidth(), grid.getHeight());

    for (int y = 0; y < grid.getHeight(); y++)
        for (int x = 0; x < grid.getWidth(); x++)
        {
            int state = rule->computeNextState(grid, x, y);
            next.set(x, y, state);
        }

    grid = next;
}

Grid &Automaton::getGrid()
{
    return grid;
}

void Automaton::printGrid()
{
    for (int y = 0; y < grid.getHeight(); y++)
    {
        for (int x = 0; x < grid.getWidth(); x++)
            std::cout << (grid.get(x, y));
        
        std::cout << std::endl;
        }
}