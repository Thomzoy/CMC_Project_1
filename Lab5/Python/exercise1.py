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
    """ Exercise 1a """

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

    # Force-length curves
    
    # Evalute for various muscle stretch
    stretch_min = muscle.L_OPT
    stretch_max = muscle.L_OPT*3
    N_stretch = 100
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
    
    activeF = np.zeros(N_stretch)
    passiveF = np.zeros(N_stretch)
    tendonF = np.zeros(N_stretch)
    lceF = np.zeros(N_stretch)

    for index_strech,stretch in enumerate(muscle_stretch):
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               stimulation=1.0,
                               muscle_length=stretch)
        activeF[index_strech] = result.active_force[-1]
        passiveF[index_strech] = result.passive_force[-1]
        tendonF[index_strech] = result.tendon_force[-1]
        lceF[index_strech] = result.l_ce[-1]
    
    plt.figure('Isometric Muscle Experiment 1a')
    
    plt.plot(lceF*100/muscle.L_OPT, activeF*100/muscle.F_MAX, label = 'Active Force')
    plt.plot(lceF*100/muscle.L_OPT, passiveF*100/muscle.F_MAX, label = 'Passive Force')
    plt.plot(lceF*100/muscle.L_OPT, tendonF*100/muscle.F_MAX, label= ' Tendon Force')

    plt.axvline(100,linewidth=1, linestyle='--', color='r')
    plt.axvline(145,linewidth=1, linestyle='--', color='r')
    
    plt.xlabel('Contractile element length [% of $L_{opt}$]')
    plt.ylabel('Force [% of $F_{max}$]')
    plt.title('Force-length curves for isometric muscle experiment (Stimulation = 1.0)')
    plt.legend()
    plt.grid()


def exercise1b():
    """ Exercise 1b """

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
    
    # Force-length curves
    
    # Evalute for various muscle stretch
    stretch_min = muscle.L_OPT*0.5
    stretch_max = muscle.L_OPT*2.8
    N_stretch = 40
    muscle_stretch = np.arange(stretch_min, stretch_max, (stretch_max-stretch_min)/N_stretch)

    # Evaluate for various muscle stimulation
    N_stim = 6
    muscle_stimulation = np.round(np.arange(N_stim)/(N_stim-1),2)
   
    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value
    
    # Set the time for integration
    t_start = 0.0
    t_stop = 0.3
    time_step = 0.001
    time = np.arange(t_start, t_stop, time_step)
    
    activeF = np.zeros(N_stretch)
    passiveF = np.zeros(N_stretch)
    tendonF = np.zeros(N_stretch)
    lceF = np.zeros(N_stretch)
    
    # Plotting
    plt.figure('Isometric Muscle Experiment 1b')
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for stim in range (len(muscle_stimulation)):
        pylog.info('Stimulation = {}'.format(stim))
        activeF = np.zeros(N_stretch)
        tendonF = np.zeros(N_stretch)       
        for stretch in range (len(muscle_stretch)):
            # Run the integration
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   stimulation=muscle_stimulation[stim],
                                   muscle_length=muscle_stretch[stretch])
            activeF[stretch] = result.active_force[-1]
            if(stim==0):
                passiveF[stretch] = result.passive_force[-1]
            tendonF[stretch] = result.tendon_force[-1]
            lceF[stretch] = result.l_ce[-1]
        
        color = colors[stim]
        plt.plot(lceF*100/muscle.L_OPT, 100*activeF/muscle.F_MAX, color, label='Active Force - Stimulation = ' + str(round(muscle_stimulation[stim], 2)))

    plt.plot(lceF*100/muscle.L_OPT, 100*passiveF/muscle.F_MAX, '+' + 'k', label='Passive Force')
    plt.title('Force-length curves for isometric muscle experiment with various muscle stimulations')
    plt.xlabel('Contractile element length [% of $L_{opt}$]')
    plt.ylabel('Force [% of $F_{max}$]')
    plt.legend()
    plt.grid()
    
    
def exercise1c():
    """ Exercice 1c """

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
    
    # Force-length curves
   
    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contractile length initial value
    
    # Set the time for integration
    t_start = 0.0
    t_stop = 0.2
    time_step = 0.001
    time = np.arange(t_start, t_stop, time_step)
    
    N_stretch = 40    
    activeF = np.zeros(N_stretch)
    passiveF = np.zeros(N_stretch)
    tendonF = np.zeros(N_stretch)
    lceF = np.zeros(N_stretch)
    
    # Evaluate for various optimal length
    l_opt_list = np.array([0.1,0.3])
    
    # Subplots grid
    n_plot = len(l_opt_list)
    n_subplot = int((np.sqrt(n_plot-1))+1)
    if (n_plot<=n_subplot*(n_subplot-1)):
        fig, axes = plt.subplots(n_subplot,n_subplot-1)
        n_subplot2 = n_subplot-1
    else:
        fig, axes = plt.subplots(n_subplot,n_subplot)
        n_subplot2 = n_subplot
    
    for i,l_opt in enumerate(l_opt_list):
    
        # Evaluate for various muscle stretch
        muscle.L_OPT = l_opt
        stretch_min = muscle.L_OPT
        stretch_max = muscle.L_OPT*3
        muscle_stretch = np.arange(stretch_min, stretch_max, (stretch_max-stretch_min)/N_stretch)
        
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
            lceF[stretch] = result.l_ce[-1]
               
        plt.subplot(n_subplot,n_subplot2,i+1)
        plt.plot(lceF, 100*activeF/muscle.F_MAX, label = 'Active Force')
        plt.plot(lceF, 100*passiveF/muscle.F_MAX, label = 'Passive Force')
        plt.plot(lceF, 100*tendonF/muscle.F_MAX, label = 'Tendon Force')
        plt.axvline(l_opt, linestyle = "--",color = "r")
        plt.xlim([l_opt-0.3,l_opt+0.3])
        plt.ylim([0,120])
        plt.xlabel('Contractile element length')
        plt.ylabel('Force [% of $F_{max}$]')
        plt.title('Optimal length = {} [m]'.format(l_opt))
        plt.legend()
        plt.grid()
        
    plt.suptitle('Force-length curves for isometric muscle experiment with various muscle optimal lengths')
    fig.tight_layout()    
    

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

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length

    # Velocity-tension curve
    
    # Evalute for various loads
    load_min = 1
    load_max = 301
    N_load = 50
    load_list = np.arange(load_min, load_max, (load_max-load_min)/N_load)

    # Evalute for Stimulation = 1.0
    stimulation = 1.0
    
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

    max_velocity = np.zeros(N_load)
    tendonF = np.zeros(N_load)
    for ind_load,load in enumerate(load_list):
        # Run the integration
        result = sys.integrate(x0=x0,
                               time=time,
                               time_step=time_step,
                               time_stabilize=time_stabilize,
                               stimulation=stimulation,
                               load=load)
        if (result.l_mtc[-1] < (sys.muscle.L_OPT + sys.muscle.L_SLACK)):
            max_velocity[ind_load] = np.max(-result.v_ce)
        else:
            max_velocity[ind_load] = np.min(-result.v_ce)
        tendonF[ind_load] = result.tendon_force[-1]
            
    # Plotting
    plt.figure('Isotonic Muscle Experiment 1d')
    v_min = np.amin(max_velocity)
    v_max = np.amax(max_velocity)
    plt.plot(max_velocity*100/-muscle.V_MAX, tendonF*100/muscle.F_MAX)
    plt.axvline(linestyle='--', color='r', linewidth=2)
    plt.text(v_min*100/-muscle.V_MAX, 20, r'lengthening', fontsize=14)
    plt.text(v_max*100/-muscle.V_MAX*1/3, 20, r'shortening', fontsize=14)
    plt.xlabel('Maximal velocity [% of $V_{max}$]')
    plt.ylabel('Tendon Force [% of $F_{max}$]')
    plt.title('Velocity-tension curve for isotonic muscle experiment (Stimulation = 1.0)')
    plt.grid()

    
def exercise1f():
    """ Exercise 1f """

    # Defination of muscles
    muscle_parameters = MuscleParameters()
    print(muscle_parameters.showParameters())

    mass_parameters = MassParameters()
    print(mass_parameters.showParameters())

    # Create muscle object
    muscle = Muscle(muscle_parameters)

    # Create mass object
    mass = Mass(mass_parameters)

    # Instatiate isotonic muscle system
    sys = IsotonicMuscleSystem()

    # Add the muscle to the system
    sys.add_muscle(muscle)

    # Add the mass to the system
    sys.add_mass(mass)

    # You can still access the muscle inside the system by doing
    # >>> sys.muscle.L_OPT # To get the muscle optimal length
    
    # Velocity-tension curve
    
    # Evalute for various loads
    load_min = 1
    load_max = 501
    N_load = 50
    load_list = np.arange(load_min, load_max, (load_max-load_min)/N_load)

    # Evalute for various muscle stimulation
    N_stim = 4
    muscle_stimulation = np.round(np.arange(N_stim)/(N_stim-1),2)
    
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

    max_velocity = np.zeros((N_stim, N_load))
    tendonF = np.zeros((N_stim, N_load))
        
    for i,stim in enumerate(muscle_stimulation):
        
        for ind_load,load in enumerate(load_list):
            # Run the integration
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   time_stabilize=time_stabilize,
                                   stimulation=stim,
                                   load=load)
            if (result.l_mtc[-1] < (sys.muscle.L_OPT + sys.muscle.L_SLACK)):
                max_velocity[i, ind_load] = np.max(-result.v_ce)
            else:
                max_velocity[i, ind_load] = np.min(-result.v_ce)
            tendonF[i, ind_load] = result.tendon_force[-1]
            
    # Plotting
    plt.figure('Isotonic Muscle Experiment 1f')
    v_min = np.amin(max_velocity)
    v_max = np.amax(max_velocity)
    for i,stim in enumerate(muscle_stimulation):
        plt.plot(max_velocity[i,:]*100/-muscle.V_MAX, tendonF[i,:]*100/muscle.F_MAX, label='Tendon Force - Stimulation = {}'.format(stim))
        plt.xlim(v_min*100/-muscle.V_MAX, v_max*100/-muscle.V_MAX)
        plt.ylim(0,200)
    plt.axvline(linestyle='--', color='r', linewidth=2)
    plt.text(v_min*100/-muscle.V_MAX*2/3, 170, r'lengthening', fontsize=16)
    plt.text(v_max*100/-muscle.V_MAX*1/8, 170, r'shortening', fontsize=16)
    plt.xlabel('Maximal velocity [% of $V_{max}$]')
    plt.ylabel('Tendon Force [% of $F_{max}$]')
    plt.title('Velocity-tension curves for isotonic muscle experiment with various muscle stimulations')
    plt.legend()
    plt.grid()


def exercise1():

    if DEFAULT["1a"] is True:
        exercise1a()
    elif DEFAULT["1b"] is True:
        exercise1b()
    elif DEFAULT["1c"] is True:
        exercise1c()  
    elif DEFAULT["1d"] is True:
        exercise1d()     
    elif DEFAULT["1f"] is True:
        exercise1f()  
    else :
        exercise1a()
        exercise1b()
        exercise1c()
        exercise1d()
        exercise1f()
        
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

