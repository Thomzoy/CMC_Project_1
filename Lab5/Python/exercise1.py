""" Lab 5 - Exercise 1 """

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

DEFAULT["label"] = [r"$\theta$ [rad]", r"$d\theta/dt$ [rad/s]"]

# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True


def exercise1a():
    """ Exercise 1a
    The goal of this exercise is to understand the relationship
    between muscle length and tension.
    Here you will re-create the isometric muscle contraction experiment.
    To do so, you will have to keep the muscle at a constant length and
    observe the force while stimulating the muscle at a constant activation."""

    # Defination of muscles
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

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length


    # Force-length curves

    # Evalute for various muscle stretch
    stretch_min = muscle.L_OPT
    stretch_max = muscle.L_OPT*2.8
    N_stretch = 40
    muscle_stretch = np.arange(stretch_min, stretch_max, (stretch_max-stretch_min)/N_stretch)

    # Evalute for various muscle stimulation
    stim_min = 0.
    stim_max = 1.5
    N_stim = 4
    #muscle_stimulation = np.arange(stim_min, stim_max, (stim_max-stim_min)/N_stim)
    muscle_stimulation = np.array([0,0.3,0.6,1])
    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001
    time = np.arange(t_start, t_stop, time_step)
    
    activeF = np.zeros(N_stretch)
    passiveF = np.zeros(N_stretch)
    tendonF = np.zeros(N_stretch)
    
    # Plotting
    n_subplot = int((np.sqrt(len(muscle_stimulation)-1))+1)

    fig, axes = plt.subplots(n_subplot,n_subplot)

    for i,stim in enumerate(muscle_stimulation):
        plt.subplot(n_subplot,n_subplot,i+1)
        
        
        for index_strech,stretch in enumerate(muscle_stretch):
            # Run the integration
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   stimulation=stim,
                                   muscle_length=stretch)
            activeF[index_strech] = result.active_force[-1]
            passiveF[index_strech] = result.passive_force[-1]
            tendonF[index_strech] = result.tendon_force[-1]
            
        #color = colors[stim]
        plt.plot(muscle_stretch*100/muscle.L_OPT, activeF*100/muscle.F_MAX, label = 'active')
        plt.plot(muscle_stretch*100/muscle.L_OPT, passiveF*100/muscle.F_MAX, label = 'passive')
        plt.plot(muscle_stretch*100/muscle.L_OPT, tendonF*100/muscle.F_MAX, label= ' tendon')
        plt.xlabel('Contractile element length [% of L_OPT]')
        plt.ylabel('Force [% of F_MAX]')
        plt.title('Stimulation : {}'.format(round(stim, 2)))
        plt.legend()
        plt.grid()
    
    plt.suptitle('Isometric Muscle Experiment')
    fig.tight_layout()
    

    
    
    # Fiber length influence
    l0 = 0.11
    l_opt_list = l0*np.arange(1,2.5,0.5)
    for l_opt in l_opt_list:
        muscle.L_OPT = l_opt
        stretch_min = muscle.L_OPT
        stretch_max = muscle.L_OPT*3
        N_stretch = 40
        muscle_stretch = np.arange(stretch_min, stretch_max, (stretch_max-stretch_min)/N_stretch)
        plt.figure()
        for stretch in range (len(muscle_stretch)):
            # Run the integration
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   stimulation=1.,
                                   muscle_length=muscle_stretch[stretch])
            activeF[stretch] = result.active_force[-1]
            passiveF[stretch] = result.passive_force[-1]
            tendonF[stretch] = result.tendon_force[-1]
        plt.plot(muscle_stretch*100/muscle.L_OPT, activeF)
        plt.plot(muscle_stretch*100/muscle.L_OPT, passiveF)
        plt.plot(muscle_stretch*100/muscle.L_OPT, tendonF)
        plt.title('l_opt = {}'.format(l_opt))
        plt.grid()


def exercise1d():
    """ Exercise 1d

    Under isotonic conditions external load is kept constant.
    A constant stimulation is applied and then suddenly the muscle
    is allowed contract. The instantaneous velocity at which the muscle
    contracts is of our interest."""

    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    pylog.warning("Isotonic muscle contraction to be implemented")

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Evalute for a single load
    load = 100.

    # Evalute for a single muscle stimulation
    muscle_stimulation = 1.

    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT,
          sys.muscle.L_OPT + sys.muscle.L_SLACK, 0.0]
    # x0[0] - -> activation
    # x0[1] - -> contractile length(l_ce)
    # x0[2] - -> position of the mass/load
    # x0[3] - -> velocity of the mass/load

    # Set the time for integration
    t_start = 0.0
    t_stop = 0.3
    time_step = 0.001
    time_stabilize = 0.2

    time = np.arange(t_start, t_stop, time_step)

    # Run the integration
    result = sys.integrate(x0=x0,
                           time=time,
                           time_step=time_step,
                           time_stabilize=time_stabilize,
                           stimulation=muscle_stimulation,
                           load=load)

    # Plotting
    plt.figure('Isometric muscle experiment')
    plt.plot(result.time, result.tendon_force)
    plt.title('Isometric muscle experiment')
    plt.xlabel('Time [s]')
    plt.ylabel('Muscle Force')
    plt.grid()


def exercise1():

    exercise1a()
    #exercise1d()

    if DEFAULT["save_figures"] is False:
        plt.show()
    else:
        figures = plt.get_figlabels()
        print(figures)
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)


if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise1()

