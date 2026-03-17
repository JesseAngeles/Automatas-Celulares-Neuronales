// engine.Automaton.h

#pragma once
#include "../core/Grid.h"
#include "../rules/Rule.h"
#include <memory>

class Automaton
{
public:
    Automaton(int width, int height, std::unique_ptr<Rule> rule);

    void Step();

    Grid& getGrid();

    void printGrid();

private:
    Grid grid;
    std::unique_ptr<Rule> rule;
};