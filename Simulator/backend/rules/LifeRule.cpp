// rules.LifeRules

#include "LifeRule.h"

int LifeRule::computeNextState(const Grid& grid, int x, int y) const {
    int neighbours = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {

            if (dx == 0 && dy == 0)
                continue;

            neighbours += grid.get(x + dx, y + dy);
        }
    }

    int current = grid.get(x, y);

    if (current == 1)
        return (neighbours == 2 || neighbours == 3);

    return (neighbours == 3);
}