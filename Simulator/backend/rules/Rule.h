// rules.Rule.h

#pragma once
#include "../core/Grid.h"

class Rule {
public:
    virtual int computeNextState(const Grid& grid, int x, int y) const = 0;
    virtual ~Rule() = default;
};