import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional

# --- Terrain constants (matching API grid codes) ---
EMPTY    = 0
SETTLE   = 1
PORT     = 2
RUIN     = 3
FOREST   = 4
MOUNTAIN = 5
OCEAN    = 10
PLAINS   = 11

PASSABLE = {EMPTY, PLAINS, FOREST}  # land types settlements can expand onto
IMPASSABLE = {OCEAN, MOUNTAIN}

# --- Tuneable parameters (these are what we'll infer) ---
@dataclass
class SimParams:
    # Growth
    food_per_forest_neighbor: float = 1.2   # food gained per adjacent forest
    food_per_plains_neighbor: float = 0.3
    consumption_per_pop: float = 1.0        # food consumed per population unit/year
    pop_growth_rate: float = 0.15           # fractional growth when food surplus
    port_build_prob: float = 0.2            # prob of gaining port if coastal + wealthy
    longship_wealth_threshold: float = 3.0  # wealth needed to build longship
    expansion_pop_threshold: float = 5.0    # pop needed to found new settlement
    expansion_prob: float = 0.1
    new_settle_inherit_frac: float = 0.3    # fraction of patron stats inherited

    # Conflict
    raid_food_threshold: float = 0.5        # food/need ratio below which raiding starts
    raid_range_land: int = 5                # tile radius for land raids
    raid_range_sea: int = 10               # additional range with longships
    conquest_prob: float = 0.3             # prob of allegiance flip on decisive win

    # Trade
    trade_range: int = 8
    trade_wealth_gain: float = 0.4
    trade_food_gain: float = 0.2
    tech_diffusion_rate: float = 0.1

    # Winter
    winter_base_severity: float = 1.0
    winter_severity_std: float = 0.4        # std of annual severity draw
    harsh_winter_prob: float = 0.1          # prob of a 2-3x severity spike
    starvation_threshold: float = -2.0      # food below this → collapse

    # Environment
    reclaim_prob_per_year: float = 0.15
    forest_regrowth_prob: float = 0.05      # ruin → forest if isolated


@dataclass
class Settlement:
    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    tech_level: float
    has_port: bool
    owner_id: int
    longships: int = 0
    alive: bool = True
    years_as_ruin: int = 0   # tracked on grid separately, but useful for reclaim logic

    def food_need(self, params: SimParams) -> float:
        return self.population * params.consumption_per_pop

    def raid_pressure(self, params: SimParams) -> float:
        ratio = self.food / max(self.food_need(params), 0.01)
        return max(0.0, 1.0 - ratio / params.raid_food_threshold)

    def raid_range(self, params: SimParams) -> int:
        base = params.raid_range_land
        return base + (params.raid_range_sea if self.longships > 0 else 0)

    def attack_strength(self) -> float:
        return self.population * self.tech_level

    def defense_strength(self) -> float:
        return self.population * self.defense * self.tech_level


class MapGenerator:
    """
    Generates terrain by seeding feature types and growing them through
    neighbor-influenced spread.  No hard structural assumptions are made
    about the shape of fjords, mountain ranges, or forests: each feature
    starts from random seed cells and expands with a probability that
    scales with the number of already-converted neighbours.

    Ocean is special: it is always present on the border and additionally
    seeded along the border to form inlets.  Mountains and forests are
    seeded in the interior and spread only onto PLAINS.
    """

    # Spread probabilities per neighbour count are determined by a single
    # per-feature parameter p: P(cell converts | k neighbours) = 1-(1-p)^k
    _OCEAN_SPREAD    = 0.40
    _MOUNTAIN_SPREAD = 0.28
    _FOREST_SPREAD   = 0.42

    def __init__(self, rng: random.Random, width: int, height: int):
        self.rng = rng
        self.width = width
        self.height = height

    def generate(self) -> tuple[np.ndarray, list["Settlement"]]:
        """Return (grid, initial_settlements)."""
        W, H = self.width, self.height
        grid = np.full((H, W), PLAINS, dtype=np.int32)

        # Hard ocean border
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OCEAN

        # Ocean inlets — seed along the border, let them grow inward
        n_inlet_seeds = self.rng.randint(4, 10)
        for _ in range(n_inlet_seeds):
            self._seed_border(grid, OCEAN)
        self._spread(grid, OCEAN, self._OCEAN_SPREAD,
                     iterations=self.rng.randint(3, 7))

        # Mountain clusters — seeded in the interior
        n_mountain_seeds = self.rng.randint(4, 10)
        for _ in range(n_mountain_seeds):
            x = self.rng.randint(3, W - 4)
            y = self.rng.randint(3, H - 4)
            grid[y, x] = MOUNTAIN
        self._spread(grid, MOUNTAIN, self._MOUNTAIN_SPREAD,
                     iterations=self.rng.randint(2, 5),
                     only_on={PLAINS})

        # Forest patches — seeded anywhere on PLAINS
        n_forest_seeds = self.rng.randint(12, 28)
        for _ in range(n_forest_seeds):
            x = self.rng.randint(1, W - 2)
            y = self.rng.randint(1, H - 2)
            if grid[y, x] == PLAINS:
                grid[y, x] = FOREST
        self._spread(grid, FOREST, self._FOREST_SPREAD,
                     iterations=self.rng.randint(1, 3),
                     only_on={PLAINS})

        settlements = self._place_settlements(grid)
        return grid, settlements

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _seed_border(self, grid: np.ndarray, terrain: int):
        """Place one seed of *terrain* at a random point on the border."""
        W, H = self.width, self.height
        edge = self.rng.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            x, y = self.rng.randint(1, W - 2), 0
        elif edge == 'bottom':
            x, y = self.rng.randint(1, W - 2), H - 1
        elif edge == 'left':
            x, y = 0, self.rng.randint(1, H - 2)
        else:
            x, y = W - 1, self.rng.randint(1, H - 2)
        grid[y, x] = terrain

    def _spread(self, grid: np.ndarray, terrain: int, spread_prob: float,
                iterations: int, only_on: set | None = None):
        """
        Iteratively expand *terrain* to neighbouring cells.
        A candidate cell with k neighbours already of *terrain* converts
        with probability  1 - (1 - spread_prob)^k.
        *only_on* restricts which current terrain types can be overwritten.
        """
        W, H = self.width, self.height
        DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        for _ in range(iterations):
            candidates: list[tuple[int, int, float]] = []
            for y in range(1, H - 1):
                for x in range(1, W - 1):
                    if grid[y, x] == terrain:
                        continue
                    if only_on is not None and grid[y, x] not in only_on:
                        continue
                    k = sum(
                        1 for dx, dy in DIRS
                        if 0 <= x + dx < W and 0 <= y + dy < H
                        and grid[y + dy, x + dx] == terrain
                    )
                    if k > 0:
                        candidates.append((x, y, 1.0 - (1.0 - spread_prob) ** k))

            for x, y, p in candidates:
                if self.rng.random() < p:
                    grid[y, x] = terrain

    def _place_settlements(self, grid: np.ndarray) -> list["Settlement"]:
        W, H = self.width, self.height
        DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

        candidates = [
            (x, y)
            for y in range(H) for x in range(W)
            if grid[y, x] in PASSABLE
        ]
        self.rng.shuffle(candidates)

        placed: list[tuple[int, int]] = []
        settlements: list[Settlement] = []
        min_dist = 6

        for x, y in candidates:
            if all(abs(x - px) + abs(y - py) >= min_dist for px, py in placed):
                is_coastal = any(
                    0 <= x + dx < W and 0 <= y + dy < H
                    and grid[y + dy, x + dx] == OCEAN
                    for dx, dy in DIRS
                )
                s = Settlement(
                    x=x, y=y,
                    population=self.rng.uniform(2.0, 4.0),
                    food=self.rng.uniform(2.0, 5.0),
                    wealth=self.rng.uniform(1.0, 3.0),
                    defense=self.rng.uniform(0.3, 0.7),
                    tech_level=1.0,
                    has_port=is_coastal and self.rng.random() < 0.5,
                    owner_id=len(placed),
                )
                grid[y, x] = PORT if s.has_port else SETTLE
                settlements.append(s)
                placed.append((x, y))
                if len(placed) >= 12:
                    break

        return settlements


class AstarWorld:
    def __init__(self, seed: int, params: SimParams, width=40, height=40):
        self.seed = seed
        self.params = params
        self.width = width
        self.height = height
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.grid = np.zeros((height, width), dtype=np.int32)
        self.settlements: list[Settlement] = []
        self.ruin_ages: dict[tuple, int] = {}  # (x,y) → years as ruin

        gen = MapGenerator(self.rng, width, height)
        self.grid, self.settlements = gen.generate()

    # ------------------------------------------------------------------
    # Simulation phases
    # ------------------------------------------------------------------

    def step_year(self):
        self._phase_growth()
        self._phase_conflict()
        self._phase_trade()
        self._phase_winter()
        self._phase_environment()

    def _phase_growth(self):
        p = self.params
        for s in self.settlements:
            if not s.alive:
                continue

            # Food production from neighbors
            food_produced = sum(
                p.food_per_forest_neighbor if self.grid[ny, nx] == FOREST
                else p.food_per_plains_neighbor if self.grid[ny, nx] == PLAINS
                else 0.0
                for nx, ny in self._neighbors(s.x, s.y)
            )
            s.food += food_produced

            # Population growth if food surplus
            if s.food > s.food_need(p):
                s.population *= (1 + p.pop_growth_rate)

            # Port development
            if not s.has_port and self._is_coastal(s.x, s.y):
                if s.wealth > 2.0 and self.rng.random() < p.port_build_prob:
                    s.has_port = True
                    self.grid[s.y, s.x] = PORT

            # Longship construction
            if s.has_port and s.longships == 0 and s.wealth >= p.longship_wealth_threshold:
                s.longships = 1
                s.wealth -= p.longship_wealth_threshold

            # Expansion: found new settlement
            if s.population >= p.expansion_pop_threshold:
                if self.rng.random() < p.expansion_prob:
                    self._found_settlement(s)

    def _phase_conflict(self):
        p = self.params
        live = [s for s in self.settlements if s.alive]
        self.rng.shuffle(live)

        for attacker in live:
            pressure = attacker.raid_pressure(p)
            if pressure <= 0:
                continue
            # Raid fires with probability proportional to pressure
            if self.rng.random() > pressure:
                continue

            rng = attacker.raid_range(p)
            targets = [
                t for t in live
                if t is not attacker
                and t.owner_id != attacker.owner_id
                and self._chebyshev(attacker, t) <= rng
            ]
            if not targets:
                continue

            # Target: weakest defense within range
            target = min(targets, key=lambda t: t.defense_strength())

            atk = attacker.attack_strength()
            dfn = target.defense_strength()
            win_prob = atk / (atk + dfn + 1e-6)

            if self.rng.random() < win_prob:
                # Raid succeeds
                loot_food   = target.food * 0.3
                loot_wealth = target.wealth * 0.2
                attacker.food   += loot_food
                attacker.wealth += loot_wealth
                target.food     -= loot_food
                target.wealth   -= loot_wealth
                target.defense  *= 0.85
                target.population *= 0.9

                # Conquest
                if target.defense < 0.2 and self.rng.random() < p.conquest_prob:
                    target.owner_id = attacker.owner_id
            else:
                # Failed raid — attacker takes losses
                attacker.population *= 0.92

    def _phase_trade(self):
        p = self.params
        ports = [s for s in self.settlements if s.alive and s.has_port]

        for i, a in enumerate(ports):
            for b in ports[i+1:]:
                if a.owner_id == b.owner_id:
                    continue
                if self._chebyshev(a, b) > p.trade_range:
                    continue
                # Trade
                a.wealth += p.trade_wealth_gain
                b.wealth += p.trade_wealth_gain
                a.food   += p.trade_food_gain
                b.food   += p.trade_food_gain
                # Tech diffusion
                avg_tech = (a.tech_level + b.tech_level) / 2
                a.tech_level += (avg_tech - a.tech_level) * p.tech_diffusion_rate
                b.tech_level += (avg_tech - b.tech_level) * p.tech_diffusion_rate

    def _phase_winter(self):
        p = self.params
        # Draw this year's severity
        severity = abs(self.np_rng.normal(p.winter_base_severity, p.winter_severity_std))
        if self.rng.random() < p.harsh_winter_prob:
            severity *= self.rng.uniform(2.0, 3.0)

        for s in self.settlements:
            if not s.alive:
                continue
            s.food -= s.food_need(self.params) * severity
            if s.food < p.starvation_threshold:
                self._collapse_settlement(s)

    def _phase_environment(self):
        p = self.params
        live = [s for s in self.settlements if s.alive]

        # Age ruins and attempt reclamation or natural decay
        for (rx, ry) in list(self.ruin_ages.keys()):
            self.ruin_ages[(rx, ry)] += 1
            age = self.ruin_ages[(rx, ry)]

            # Check for nearby patron
            patrons = [
                s for s in live
                if self._chebyshev_xy(s.x, s.y, rx, ry) <= 4
                and s.population > p.expansion_pop_threshold * 0.7
            ]
            if patrons and self.rng.random() < p.reclaim_prob_per_year:
                patron = max(patrons, key=lambda s: s.population)
                self._reclaim_ruin(rx, ry, patron)
            elif age > 5 and self.rng.random() < p.forest_regrowth_prob:
                self.grid[ry, rx] = FOREST
                del self.ruin_ages[(rx, ry)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collapse_settlement(self, s: Settlement):
        s.alive = False
        self.grid[s.y, s.x] = RUIN
        self.ruin_ages[(s.x, s.y)] = 0

        # Disperse population to nearby friendlies
        neighbors = [
            t for t in self.settlements
            if t.alive and t.owner_id == s.owner_id
            and self._chebyshev(s, t) <= 6
        ]
        if neighbors:
            share = s.population * 0.5 / len(neighbors)
            for t in neighbors:
                t.population += share

    def _found_settlement(self, patron: Settlement):
        candidates = [
            (x, y)
            for nx, ny in self._neighbors_radius(patron.x, patron.y, 3)
            for x, y in [(nx, ny)]
            if self.grid[y, x] in PASSABLE
            and not any(s.x == x and s.y == y for s in self.settlements)
        ]
        if not candidates:
            return
        x, y = self.rng.choice(candidates)
        p = self.params
        f = p.new_settle_inherit_frac
        is_coastal = self._is_coastal(x, y)
        s = Settlement(
            x=x, y=y,
            population=patron.population * f,
            food=patron.food * f,
            wealth=patron.wealth * f,
            defense=patron.defense,
            tech_level=patron.tech_level,
            has_port=is_coastal and self.rng.random() < 0.4,
            owner_id=patron.owner_id,
        )
        patron.population *= (1 - f * 0.5)
        self.grid[y, x] = PORT if s.has_port else SETTLE
        self.settlements.append(s)

    def _reclaim_ruin(self, rx, ry, patron: Settlement):
        p = self.params
        f = p.new_settle_inherit_frac
        is_coastal = self._is_coastal(rx, ry)
        s = Settlement(
            x=rx, y=ry,
            population=patron.population * f * 0.5,
            food=patron.food * f,
            wealth=patron.wealth * f * 0.5,
            defense=patron.defense * 0.7,
            tech_level=patron.tech_level,
            has_port=is_coastal,
            owner_id=patron.owner_id,
        )
        self.grid[ry, rx] = PORT if s.has_port else SETTLE
        self.settlements.append(s)
        del self.ruin_ages[(rx, ry)]

    def _neighbors(self, x, y):
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                yield nx, ny

    def _neighbors_radius(self, x, y, r):
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    yield nx, ny

    def _is_coastal(self, x, y):
        return any(self.grid[ny, nx] == OCEAN for nx, ny in self._neighbors(x, y))

    def _chebyshev(self, a: Settlement, b: Settlement) -> int:
        return max(abs(a.x - b.x), abs(a.y - b.y))

    def _chebyshev_xy(self, ax, ay, bx, by) -> int:
        return max(abs(ax - bx), abs(ay - by))

    # ------------------------------------------------------------------
    # Monte Carlo output — matches the API's ground truth format
    # ------------------------------------------------------------------

    def to_prediction_tensor(self) -> np.ndarray:
        """Single-run hard prediction (one-hot). For MC averaging, run many times."""
        CLASS_MAP = {OCEAN:0, PLAINS:0, EMPTY:0, SETTLE:1, PORT:2, RUIN:3, FOREST:4, MOUNTAIN:5}
        tensor = np.zeros((self.height, self.width, 6), dtype=np.float32)
        for y in range(self.height):
            for x in range(self.width):
                cls = CLASS_MAP.get(self.grid[y, x], 0)
                tensor[y, x, cls] = 1.0
        return tensor


def run_monte_carlo(seed: int, params: SimParams, n_runs: int = 200,
                    width=40, height=40, years=50) -> np.ndarray:
    """
    Run the simulator n_runs times and average the results into a
    probability tensor — matching the API's ground truth computation.
    """
    accum = np.zeros((height, width, 6), dtype=np.float64)
    for _ in range(n_runs):
        world = AstarWorld(seed, params, width, height)
        for _ in range(years):
            world.step_year()
        accum += world.to_prediction_tensor()
    return (accum / n_runs).astype(np.float32)
