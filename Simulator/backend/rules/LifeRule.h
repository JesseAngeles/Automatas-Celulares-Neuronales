// rules.LifeRule.h

#pragma once
#include "Rule.h"

class LifeRule: public Rule {
public:
    int computeNextState(const Grid& grid, int x, int y) const override;
};
