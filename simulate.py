'''
Runs the simulation study and saves raw results to results/raw/
'''
import logging

from scenarios import SCENARIOS
from simulation import SimulationStudy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

study = SimulationStudy(SCENARIOS)
study.run()