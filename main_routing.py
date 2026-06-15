import random
import copy
from graph_utils.upload_graph import make_grid_graph, load_map_graph_xml
from extended_time_graph import TimeExpandedGraph
from Simulator import Simulator
from MAPD_solver import MAPD_solver
import matplotlib.pyplot as plt
import networkx as nx
from utils import *
import os

"""
MAPD simulation on a map graph, dynamic missions.
Setup:
  - graph loaded from data/graph_xml/test_graph_20nodes.xml
  - 4 AMRs with initial positions [1, 4, 11, 3]
  - AMR1 starts idle (no initial mission) — stays at node 4 until step 2.
  - AMR0, AMR2, AMR3 get initial missions with intermediate waypoints
  - AMR0 and AMR2 receive additional dynamic missions at steps 3 and 4

Flow:
  - subsequent replans (triggered by dynamic_mission_targets) using MAPD
  - at each simulation step: check_progress_conflicts verifies the current
    _progress is vertex/edge-conflict free, then sim.step() advances time
"""

random.seed(42)

# grafo 
# G = make_grid_graph(rows=4, cols=4, step=1.0)

file_path = os.path.join("data", "graph_xml", "test_graph_20nodes.xml")
# create graph
G = load_map_graph_xml(file_path)

# posizioni iniziali degli AMR 
initial_positions = [1, 4, 11, 3 ]   # un nodo per agente

#simulatore
sim = Simulator(graph=G, initial_positions=initial_positions)


missions = [  # missioni iniziali
    (0, [sim.amr_positions[0], 3, 7, 5]),
    (2, [sim.amr_positions[2], 10, 2]),
    (3, [sim.amr_positions[3], 9, 3])
]

dynamic_mission_targets = { # quando sono aggiunte missioni
    2: [(1, [6])],
    3: [(0, [9])],
    4: [(2, [9,3])]
}

print("\n--- Already cheduled paths at beginning (_progress): ---")
for agent_id, path in sim._progress.items():
    print(f"  AMR {agent_id}: {path}")


# Call MAPD

'''
Esempi di chiamata dei solver PP, CBS e BCBS (algoritmi MAPF) che lavorano internamente al MAPD
mapd = MAPD_solver(sim, solver_name="PP")
# oppure
mapd = MAPD_solver(sim, solver_name="CBS")
# oppure con parametri extra per BCBS (es. w=1.5)
mapd = MAPD_solver(sim,solver_name="BCBS",solver_kwargs={"w_l": 1.5,"w_h": 1.2,"conflict_heuristic_low_l": "h3","conflict_heuristic_high_l": "h3",})
'''

mapd = MAPD_solver(sim, solver_name="BCBS", solver_kwargs={"w_l": 1.5,"w_h": 1.2,"conflict_heuristic_low_l": "h3","conflict_heuristic_high_l": "h3"})
success = mapd.compute_paths(missions)

if not success:
    print("MAPD failed — no solution found")
else:
    print("TIME 0")
    print("\n Starting + Waypoints new mission")
    for amr_id, wp in missions:
        print(f"  AMR {amr_id}: {wp}")
    print("\n--- Scheduled paths (_progress) ---")
    for agent_id, path in sim._progress.items():
        print(f"  AMR {agent_id}: {path}")
    print()

step_num, working_status = sim.step()
# _, working_status = sim.step(paths)   -> rimosso caricamento paths , il salvataggio del path viene fatto direttamente in MAPD che ha accesso a sim (il salvataggio viene fatto iterativamente ad ogni waypoint)
# e i mapf in sequenza usano il progress aggiornato (volendo se si vuole portare fuori si costruisce una copia di progress interna)
i = 0

while any(working_status):

    ok, conflicts = check_progress_conflicts(sim)
    if not ok:
        print("CONFLICTS FOUND:")
        for c in conflicts:
            print(f"  {c}")
    else:
        print("No conflicts (vertex or swap edges) in scheduled paths.")


    print("TIME", step_num)
    print(f"  step {step_num:3d} — positions: {sim.amr_positions}")

    print("\n--- Scheduled paths (_progress) ---")
    for agent_id, path in sim._progress.items():
        print(f"  AMR {agent_id}: {path}")
    print()

    # DYNAMIC MISSIONS
    if step_num in dynamic_mission_targets:
        '''
        new_missions = [
            (amr_id, [env.amr_positions[amr_id]] + targets)
            for amr_id, targets in dynamic_mission_targets[env.step_num]
        ]
        '''
        # sostituita con codice di seguito (la partenza dovrebbe essere dall'ultimo nodo schedulato quindi sim._progress[amr_id][-1])

        new_missions = []
        for amr_id, targets in dynamic_mission_targets[step_num]:
            if amr_id not in sim._progress or len(sim._progress[amr_id]) == 0:
                starting_node = sim.amr_positions[amr_id]
            else:
                starting_node = sim._progress[amr_id][-1]
            new_mission = ( amr_id,[starting_node] + targets)
            new_missions.append(new_mission)

        # Call MAPD after new missions arrive
        mapd = MAPD_solver(sim, solver_name="PP")
        success = mapd.compute_paths(new_missions)

        if not success:
            print(f"MAPD replanning failed at step {step_num}")
            break

        print("\n--- Starting + Waypoints ---")
        for amr_id, wp in new_missions:
            print(f"  AMR {amr_id}: {wp}")

        print("\n--- Scheduled paths (_progress) ---")
        for agent_id, path in sim._progress.items():
            print(f"  AMR {agent_id}: {path}")
        print()

        step_num, working_status = sim.step()
    else:
        step_num, working_status = sim.step()

    i += 1

print("\n--- Final positions ---")
for i, pos in enumerate(sim.amr_positions):
    print(f"  AMR {i}: node {pos}")


