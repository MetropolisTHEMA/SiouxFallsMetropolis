import os
import json

import numpy as np
import pandas as pd

# Path to the directory where the output files should be stored.
OUTPUT_DIR = "./output/1"
# Vehicle length in meters.
VEHICLE_LENGTH = 7.0
# Vehicle passenger-car equivalent.
VEHICLE_PCE = 1.0
# Simulation period.
PERIOD = [5.0 * 3600, 10.0 * 3600.0]
# Capacity per lane, used to compute the number of lanes on each edge (the total capacity of each
# edge is given by the input files and fixed).
CAPACITY_PER_LANE = 2000
# Edges' speed limit (set to 60 km/h so that free-flow travel time in minutes is equal to edge
# length in kilometers, just like in the input data).
SPEED_LIMIT = 60
# If `True`, vehicles can overtake each other at the edge's exit bottleneck (only if they have a
# different downstream edge). Recommanded value is: `True`.
ENABLE_OVERTAKING = True
# Value of time, in euros per hour.
ALPHA = 10.0
# Desired arrival time (in seconds after midnight).
T_STAR = 8.0 * 3600.0
# Early arrival penalty, in euros per hour.
BETA = 5.0
# Late arrival penalty, in euros per hour.
GAMMA = 20.0
# Desired arrival time window, in seconds.
DELTA = 0.0
# Variance of the errors in the departure-time choice model.
MU = 2.0
# Simulation parameters.
PARAMETERS = {
    "period": PERIOD,
    "learning_model": {
        "type": "Exponential",
        "value": {
            "alpha": 0.99,
        },
    },
    "init_iteration_counter": 1,
    "stopping_criteria": [
        {
            "type": "MaxIteration",
            "value": 100,
        },
    ],
    "update_ratio": 1.0,
    "random_seed": 13081996,
    "network": {
        "road_network": {
            "recording_interval": 300.0,
            "spillback": True,
            "max_pending_duration": 30.0,
        }
    },
}

print("Reading edges")
edges = pd.read_csv("data/SiouxFalls_net.tntp", sep="\t", skiprows=7)

print("Creating Metropolis road network")
metro_edges = list()
for i, row in edges.iterrows():
    nb_lanes = int(1 + row["capacity"] // CAPACITY_PER_LANE)
    edge = [
        row["init_node"] - 1,
        row["term_node"] - 1,
        {
            "id": i,
            "base_speed": SPEED_LIMIT / 3.6,
            "length": float(row["length"]) * 1000.0,
            "lanes": nb_lanes,
            "speed_density": {
                "type": "FreeFlow",
            },
            "bottleneck_flow": row["capacity"] / (nb_lanes * 3600),
            "overtaking": ENABLE_OVERTAKING,
        },
    ]
    metro_edges.append(edge)

graph = {
    "edges": metro_edges,
}

vehicles = [
    {
        "length": VEHICLE_LENGTH,
        "pce": VEHICLE_PCE,
        "speed_function": {
            "type": "Base",
        },
    }
]

road_network = {
    "graph": graph,
    "vehicles": vehicles,
}

print("Reading trips")
od_matrix = pd.read_csv("data/SiouxFalls_od.csv", dtype={"Ton": int})

print("Generating trips")

agents = list()
us = np.random.uniform(0, 1, size=int(od_matrix["Ton"].sum()))
i = 0
for _, od_pair in od_matrix.iterrows():
    for _ in range(od_pair["Ton"]):
        departure_time_model = {
            "type": "ContinuousChoice",
            "value": {
                "period": PERIOD,
                "choice_model": {
                    "type": "Logit",
                    "value": {
                        "u": float(us[i]),
                        "mu": MU,
                    },
                },
            },
        }
        schedule_utility = {
            "type": "AlphaBetaGamma",
            "value": {
                "beta": BETA / 3600.0,
                "gamma": GAMMA / 3600.0,
                "t_star_high": T_STAR + DELTA / 2.0,
                "t_star_low": T_STAR - DELTA / 2.0,
            },
        }
        leg = {
            "class": {
                "type": "Road",
                "value": {
                    "origin": int(od_pair["O"] - 1),
                    "destination": int(od_pair["D"] - 1),
                    "vehicle": 0,
                },
            },
        }
        car_mode = {
            "type": "Trip",
            "value": {
                "total_travel_utility": {
                    "type": "Polynomial",
                    "value": {
                        "b": -ALPHA / 3600.0,
                    },
                },
                "departure_time_model": departure_time_model,
                "destination_schedule_utility": schedule_utility,
                "legs": [leg],
            },
        }
        agent = {
            "id": i,
            "modes": [car_mode],
        }
        agents.append(agent)
        i += 1

print("Writing data...")
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
with open(os.path.join(OUTPUT_DIR, "network.json"), "w") as f:
    f.write(json.dumps(road_network))
with open(os.path.join(OUTPUT_DIR, "agents.json"), "w") as f:
    f.write(json.dumps(agents))
with open(os.path.join(OUTPUT_DIR, "parameters.json"), "w") as f:
    f.write(json.dumps(PARAMETERS))
