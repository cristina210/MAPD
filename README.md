# MAPD — Multi-Agent Pickup and Delivery
Framework for planning collision-free, multi-waypoint routes for a fleet of AMRs (Autonomous Mobile Robots) moving on a graph. It is built in two layers:

- a **MAPF** layer (Multi-Agent Path Finding) — three solvers (`PP`, `CBS`, `BCBS`) that compute conflict-free paths on a Time-Expanded Graph (TEG),
  supporting agents with different start times and external (already occupied) obstacles;
- a **MAPD** layer on top of it — `MAPD_solver`, which turns a list of per-agent waypoint sequences into a sequence of MAPF calls over a sliding
  time window, and writes the resulting paths into a `Simulator`.

**Main: `main_routing.py`.** It builds a graph, creates a `Simulator`, assigns initial missions, runs `MAPD_solver`, and then steps the simulation
forward, injecting new missions at given timesteps.

---

## Project structure

```
MAPD/
├── main_routing.py               # MAIN — full MAPD simulation example
├── Simulator.py                  # Simulator: AMR positions + _progress (scheduled future nodes)
├── MAPD_solver.py                # MAPD_solver — windowed/rolling-horizon replanning on top of MAPF
├── extended_time_graph.py        # TimeExpandedGraph (TEG)
├── utils.py                       # compute_T_min, verify_solution_with_constraints, check_progress_conflicts
├── utils_per_result.py           # helpers for the comparison tests
│
├── instance/
│   └── Network_graph.py          # NetworkGraph (nx.DiGraph subclass) + NodeType enum
│
├── graph_utils/
│   ├── upload_graph.py           # make_grid_graph, load_map_graph_{xml,json,yaml}
│   ├── graph_visualization.py    # plot_graph, print_expanded_graph
│   └── time_graph_builder.py     # low-level TEG construction (id remapping, swap pairs, constraints)
│
├── fleet_utils/
│   ├── fleet.py                  # Agent, Fleet (just use in tests)
│   └── upload_fleet.py           # make_random_fleet — used by the MAPF comparison test
│
├── shortest_path_algorithm/
│   └── A_star.py                 # a_star, a_star_with_focal_search, heuristics
│
├── MAPF_algorithm/
│   ├── heuristicsBCBS.py          # conflict heuristics (h1/h2/h3) for BCBS, high- and low-level
│   └── solvers/
│       ├── PP_mapd.py             # PrioritizedPlanner_mapd
│       ├── CBS_mapd.py            # CBSSolver_mapd
│       └── BCBS_mapd.py           # BCBSSolver_mapd
│
├── data/
│   ├── graph_xml/                # map graphs in MapCfg XML format (test_graph_20nodes.xml, ...)
│   ├── graph_json/                # map graphs in JSON format
│   └── toy_map.yaml               # map graph in YAML format
│
├── tests/
│   ├── MAPD_only_one_call.py                          # MAPD_solver: PP vs CBS vs BCBS on random instances
│   └── MAPF_algorithm_with_obstacles_and_asynchronous.py  # MAPF solvers: async starts + external obstacles
│
└── result/                        # output of the two test scripts above (text reports)
```

---

## Core concepts

### Time-Expanded Graph (TEG)

`TimeExpandedGraph` (`extended_time_graph.py`) replicates the spatial graph
`G` across `T` timesteps. Every spatial node `original_id` (given in input) is first remapped
to a consecutive `internal_id` (0..N-1), then expanded into `T` time-indexed
copies via

```
expanded_id = internal_id * T + t
```

Two edge types connect them: **wait** edges `(node, t) -> (node, t+1)` and
**move** edges `(u, t) -> (v, t+1)`. Vertex/edge constraints remove nodes/edges from this graph before planning. `swap_pairs` records, for every bidirectional edge, the "mirrored" arc used to forbid two agents from swapping positions in one step.

### MAPF solvers (`MAPF_algorithm/solvers/*_mapd.py`)

All three solvers share the same interface:

```python
solver = SolverClass(teg=teg, diz_constr_vertex=vertex_constr, diz_constr_edge=edge_constr, ...)
paths, success = solver.plan(diz_start_and_goal, diz_time_start)
```

- `diz_start_and_goal`: `{agent_id: (start_node, goal_node)}`, original node ids.
- `diz_time_start`: `{agent_id: start_t}` — agents may start at different
  timesteps of the same TEG.
- `diz_constr_vertex` / `diz_constr_edge`: `{agent_id: set(...)}` — external,
  pre-existing obstacles (expanded node ids / expanded edge pairs).
- `paths`: `{agent_id: [expanded_node_ids...]}` on success.

**PrioritizedPlanner_mapd (`PP`)** — plans agents one at a time; each path
becomes a constraint for the remaining agents. Fast, but not complete nor
optimal.

**CBSSolver_mapd (`CBS`)** — Conflict-Based Search: optimal (sum-of-costs)
and complete. Splits on the first detected conflict, exploring a constraint
tree with best-first search.

**BCBSSolver_mapd (`BCBS`)** — bounded-suboptimal CBS using focal search at
both levels (`w_l` for the low-level A*, `w_h` for the high-level constraint
tree). `w_l=w_h=1` is equivalent to CBS; larger values trade optimality for
speed. The conflict heuristics (`h1`, `h2`, `h3`, see `MAPF_algorithm/heuristicsBCBS.py`) guide focal search toward fewer-conflict solutions.

`MAPD_solver.SOLVER_REGISTRY` pick the solver by name plus extra kwargs:

```python
mapd = MAPD_solver(sim, solver_name="BCBS",
                    solver_kwargs={"w_l": 1.5, "w_h": 1.2,
                                   "conflict_heuristic_low_l": "h3",
                                   "conflict_heuristic_high_l": "h3"})
```

### From MAPF to MAPD: windowed replanning (`MAPD_solver.py`)

`MAPD_solver.compute_paths(missions)` takes a list of `(agent_id, [current_position, waypoint_1, waypoint_2, ...])` and plans a
path through every waypoint for every listed agent, writing the result into `sim._progress`.

Only the agents listed in `missions` are (re)planned. Every other agent's `_progress` is treated as a **fixed trajectory** and injected as external
vertex/edge constraints (`_extract_constr`) for the agents being planned — including the *old, already-committed* portion of an agent that is also
being replanned but already has a non-empty `_progress`.

Internally, this is a loop of MAPF sub-problems over a sliding time window:

1. Build a TEG with a horizon `T` estimated from current shortest-path distances (`_makespan_shortest_path`).
2. Run the chosen MAPF solver once for all agents still being planned, jointly, toward their current intermediate waypoint. Agents may have
   different shifted start times within this TEG (`diz_time_start_mapd_shifted_for_agent`), where `t=0` corresponds to the global time `t_min`.
3. Let `t_min_local` be the earliest TEG time at which any planned agent reaches its waypoint. Commit, for every agent, only the path prefix up to
   `t_min_local` into `_progress` (`steps_committed`) — this is the rolling-horizon/windowed part: full joint plans are computed, but only the
   part up to the first waypoint event is taken as final.
4. Advance `t_min`, update each agent's (start, goal) pair — next waypoint for agents that arrived, same goal with a new start otherwise — and
   recompute each agent's shifted start time for the next iteration.
5. Repeat until every agent in `missions` has visited all its waypoints.


### Simulator (`Simulator.py`)

- `sim.amr_positions[agent_id]` — current node of each agent (index = agent id).
- `sim._progress[agent_id]` — list of **future** nodes already scheduled for
  that agent (does **not** include the current position).
- `sim.step()` — advances time by one step: pops the first element of each
  agent's `_progress` into `amr_positions`.

---

## Using `MAPD_solver` directly

```python
from graph_utils.upload_graph import make_grid_graph
from Simulator import Simulator
from MAPD_solver import MAPD_solver

G = make_grid_graph(rows=4, cols=4, step=1.0)
sim = Simulator(graph=G, initial_positions=[1, 5, 11, 3])

missions = [
    (0, [sim.amr_positions[0], 3, 7, 5]),   # AMR 0: current pos -> 3 -> 7 -> 5
    (2, [sim.amr_positions[2], 10, 2]),
]

mapd = MAPD_solver(sim, solver_name="CBS")
success = mapd.compute_paths(missions)

# sim._progress now contains the planned future nodes for AMR 0 and 2
```

To advance the simulation and inject new missions over time, see the main
loop in `main_routing.py` (it re-derives each agent's current start node from
`sim._progress[agent_id][-1]` if it has one, otherwise `sim.amr_positions`).

---

## Loading graphs

`graph_utils/upload_graph.py`:

- `make_grid_graph(rows, cols, step)` — synthetic grid, consecutive node ids `0..rows*cols-1` use for testing.
- `load_map_graph_xml(path)` —  `data/graph_xml/test_graph_20nodes.xml`.
- `load_map_graph_json(path)` 
- `load_map_graph_yaml(path, level="L1")`

All of them return a `NetworkGraph` (`instance/Network_graph.py`).

TEG construction (`build_id_remapping`) only requires node ids to be comparable — it remaps them to `0..N-1` internally. So any input nodes id are okay

---

## Tests

Both scripts in `tests/` add the project root to `sys.path` and can be run
directly (`python tests/<name>.py`); they write a text report to `result/`.

- **`MAPD_only_one_call.py`** — for `N_ITER` random instances (random initial
  positions, random pre-existing `_progress`, random multi-waypoint
  missions), runs `MAPD_solver` once with each of PP/CBS/BCBS and checks
  success, conflict-freedom of `_progress` (`check_progress_conflicts`),
  whether all waypoints were reached in order, and total path cost.

- **`MAPF_algorithm_with_obstacles_and_asynchronous.py`** — for random
  fleets with asynchronous start times (`make_random_fleet`,
  `mapf_sincrono=False`) plus injected external vertex/edge obstacles, runs
  `CBSSolver_mapd`, `BCBSSolver_mapd`, `PrioritizedPlanner_mapd` on the same
  instance and checks success, conflict/constraint-freedom
  (`verify_solution_with_constraints`), and compares sum-of-costs (CBS is
  optimal, so PP/BCBS should never beat it).

---

## Known issues / work in progress

Still open (from the `ANCORA DA VEDERE` notes in `MAPD_solver.py` plus a
couple of things noticed while reading the code — none of these block
`main_routing.py`, they matter for edge cases / future extensions):

- **Idle AMRs as obstacles** — an agent present only in `sim.amr_positions`
  (no `_progress`, not in the current `missions`) is not yet injected as a
  constraint in `_extract_constr`. If another agent's already-committed path
  passes through that idle AMR's node, and the idle AMR later receives a
  mission starting from that same node, the new mission may be hard or
  impossible to plan (its start node can't itself be constrained). (Questo potrebbe essere il motivo per cui alcuni conflitti permangono nei test)
- Understand crash situation in MAPD

---

## Dependencies

```bash
pip install networkx matplotlib pyyaml
```
