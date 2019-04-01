# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:21:50 2019

@author: Dell
"""

import matplotlib.pyplot as plt
import numpy as np

import cmc_pylog as pylog
from muscle import Muscle
from mass import Mass
from cmcpack import DEFAULT, parse_args
from cmcpack.plot import save_figure
from system_parameters import MuscleParameters, MassParameters
from isometric_muscle_system import IsometricMuscleSystem
from isotonic_muscle_system import IsotonicMuscleSystem

parameters = MuscleParameters()
pylog.warning("Loading default muscle parameters")
pylog.info(parameters.showParameters())
pylog.info("Use the parameters object to change the muscle parameters")

# Create muscle object
muscle = Muscle(parameters)

# Instatiate isometric muscle system
sys = IsometricMuscleSystem()

# Add the muscle to the system
sys.add_muscle(muscle)

stretch_min = muscle.L_OPT
stretch_max = muscle.L_OPT*3
N_stretch = 40
muscle_stretch = np.arange(stretch_min, stretch_max, (stretch_max-stretch_min)/N_stretch)

# Set the initial condition
x0 = [0.0, sys.muscle.L_OPT]
# x0[0] --> muscle stimulation intial value
# x0[1] --> muscle contracticle length initial value

# Set the time for integration
t_start = 0.0
t_stop = 0.2
time_step = 0.001
time = np.arange(t_start, t_stop, time_step)

result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               stimulation=1,
                               muscle_length=2.2*muscle.L_OPT)
print(0.5*muscle.L_OPT)

plt.plot(result.time,result.l_ce)
plt.figure()
plt.plot(result.time,result.active_force)
print(result.active_force[-1])
