import hashlib
import json
import logging
from pathlib import Path
import platform
from time import perf_counter
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy
import statsmodels
from threadpoolctl import threadpool_limits

from dgps import DGP
from methods import Methods


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

    def __init__(self, scenarios, n_sim=1000, seed=607,
                 out_dir="results/raw"):
        self.scenarios, self.n_sim, self.seed = scenarios, n_sim, seed
        self.out_dir = Path(out_dir)

        # Everything that defines the study; any change gives a new study_id
        src_dir = Path(__file__).resolve().parent
        source_files = sorted(src_dir.glob("*.py"))
        self.config = {
            "scenarios": scenarios,
            "n_sim": n_sim,
            "seed": seed,
            "rng": {"generator": "PCG64",
                    "seed_sequence": "[seed, *SHA256_words]",
                    "label": "compact JSON [scenario_id, rep, task]",
                    "rule": "one population per scenario: rep=0, task='dgp'; "
                            "the same stream builds the population and then the randomizations"},
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in source_files},
            "environment": {"python": platform.python_version(),
                            "numpy": np.__version__, "pandas": pd.__version__,
                            "scipy": scipy.__version__,
                            "statsmodels": statsmodels.__version__},
        }
        fingerprint = json.dumps(self.config, sort_keys=True).encode()
        self.study_id = "study_" + hashlib.sha256(fingerprint).hexdigest()[:12]
        self.study_dir = self.out_dir / self.study_id

    def run_scenario(self, scenario_id, settings):
        '''
        Run a scenario and output results
        '''
        # Check if scenario has been run
        checkpoint = self.study_dir / f"{scenario_id}.csv"
        if checkpoint.exists():
            logging.info("Reusing %s", scenario_id)
            saved = pd.read_csv(checkpoint)
            saved["failure_reason"] = saved["failure_reason"].fillna("")
            return saved
        
        # Generate data for experiment
        rng = rng_for(self.seed, scenario_id, 0, "dgp")

        dgp = DGP(n_sample=settings['n_sample'],
                  rng = rng,
                  n_sim = self.n_sim,
                  pi_treat = settings['pi_treat'],
                  cut_control = settings['cut_control'],
                  cut_treat = settings['settings'],
                  observed_covariate = settings['observed_covariate']
                  )

        repetition_data = dgp.generate_repetitions()
        true_delta = dgp.true_delta()

        # Perform methods on experimental data
        rows = []
        for repetition in tqdm(repetition_data, desc=scenario_id, unit="rep"):
            start = perf_counter()
            failure_reason = ""

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                try:
                    methods = Methods(repetition['trt'], repetition['Y_observed'])
                    estimates = methods.run_methods()
                except Exception as error:
                    estimates = {"beta1_hat": np.nan, "beta2_hat": np.nan,
                                 "beta3_hat": np.nan, "plug_in": np.nan,
                                 "itt": np.nan}
                    failure_reason = type(error).__name__

            if failure_reason == "" and len(caught) > 0:
                failure_reason = "warning: " + type(caught[0].message).__name__

            row = {
                "scenario_id": scenario_id,
                "sim_index": repetition["sim_index"],
                "n_sample": settings["n_sample"],
                "pi_treat": settings["pi_treat"],
                "true_delta": true_delta,
                "failure_reason": failure_reason,
                "seconds": perf_counter() - start,
            }
            row.update(estimates)
            rows.append(row)

        frame = pd.DataFrame(rows)
        save_csv(frame, checkpoint)
        logging.info("Completed %s (%d reps)", scenario_id, self.n_sim)
        return frame

    def run(self):
        '''
        Saves configuration, runs all scenarios, exports merged raw results
        '''
        # Save configuration
        metadata = {"study_id": self.study_id, **self.config}
        save_json(metadata, self.study_dir / "metadata.json")

        # Run scenarios
        frames = []
        with threadpool_limits(limits=1):
            for scenario_id, settings in self.scenarios.items():
                frames.append(self.run_scenario(scenario_id, settings))

        # Combine results and export
        results = pd.concat(frames, ignore_index=True)
        save_csv(results, self.study_dir / "runs.csv")
        save_csv(results, self.out_dir / "runs.csv")

        design = pd.DataFrame([{"scenario_id": scenario_id, **settings}
                               for scenario_id, settings in self.scenarios.items()])
        save_csv(design, self.out_dir / "design.csv")
        save_json(metadata, self.out_dir / "metadata.json")

        return results