"""Predeclared nested sensor layouts, selected without concentration data."""
import numpy as np

COUNTS = [4, 8, 16]
SEEDS = [7, 42, 123]
NOISES = [0., .02, .05]


def layout_nodes(layout):
    if layout == 0:
        # Prefix of eight equals the published eight sensors as a set.
        order = [8, 112, 270, 375, 60, 165, 217, 322]
        while len(order) < 16:
            candidates = np.arange(8, 376)
            distance = np.min(np.abs(candidates[:, None]-np.array(order)), axis=1)
            order.append(int(candidates[np.argmax(distance)]))
        return np.array(order), np.full(16, 63)
    # Three spatially distributed layouts: seeded candidate pool, physical-distance
    # farthest-point selection. No source locations, concentrations, or errors used.
    rng = np.random.default_rng(1700+layout)
    pairs = np.unique(np.column_stack([rng.integers(8, 376, 4096), rng.integers(8, 120, 4096)]), axis=0)
    coords = pairs*np.array([9/383, 3/127])
    chosen = [int(rng.integers(len(pairs)))]
    for _ in range(15):
        dist = np.min(np.sum((coords[:, None]-coords[chosen])**2, axis=2), axis=1)
        chosen.append(int(np.argmax(dist)))
    return pairs[chosen, 0], pairs[chosen, 1]


def validate():
    for layout in range(4):
        ix, iy = layout_nodes(layout)
        assert len(set(zip(ix, iy))) == 16
        assert np.all((ix >= 0) & (ix < 384) & (iy >= 0) & (iy < 128))
        for n in COUNTS:
            assert len(ix[:n]) == n
    assert set(layout_nodes(0)[0][:8]) == {8,60,112,165,217,270,322,375}


if __name__ == '__main__':
    validate()
    print('Four nested layouts verified; line n=8 matches original sensor set.')
