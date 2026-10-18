import hashlib
import json
import logging
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from dgps.py import DGP


def rng_for(seed, scenario_id, rep, task):
    label = json.dumps([scenario_id, rep, task], separators=(",", ":"))
    digest = hashlib.sha256(label.encode()).digest()[:16]
    words = np.frombuffer(digest, dtype="<u4").tolist()
    return np.random.default_rng(np.random.SeedSequence([seed, *words]))

def save_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)

def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

class SimulationStudy:
    '''
    Using the scenario dictionary, run the simulation
    '''

    def __init__(self, scenarios, seed=607,
                 out_dir="results/raw"):
        self.scenarios, self.seed = scenarios, seed
        self.out_dir = Path(out_dir)

    def run_scenario(self, scenario_id, settings):
        '''
        Run a scenario and output results
        '''
        experiment_data = 

    