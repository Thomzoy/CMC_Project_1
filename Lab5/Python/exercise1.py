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
        
    #color = colors[stim]
    """
    plt.plot(muscle_stretch*100/muscle.L_OPT, activeF*100/muscle.F_MAX, label = 'Active')
    plt.plot(muscle_stretch*100/muscle.L_OPT, passiveF*100/muscle.F_MAX, label = 'Passive')
    plt.plot(muscle_stretch*100/muscle.L_OPT, tendonF*100/muscle.F_MAX, label= ' Tendon')
    """
    plt.plot(lceF*100/muscle.L_OPT, activeF*100/muscle.F_MAX, label = 'Active')
    plt.plot(lceF*100/muscle.L_OPT, passiveF*100/muscle.F_MAX, label = 'Passive')
    plt.plot(lceF*100/muscle.L_OPT, tendonF*100/muscle.F_MAX, label= ' Tendon')

    plt.axvline(100,linewidth=1, linestyle='--', color='r')
    plt.axvline(145,linewidth=1, linestyle='--', color='r')
    plt.xlabel('Contractile element length [% of $L_{opt}$]')
    plt.ylabel('Force [% of $F_{max}$]')
    plt.title('Force-length curves for isometric muscle experiment (Stimulation : 1.0)')
    plt.legend()
    plt.grid()


def exercise1b_color():
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
    stretch_min = muscle.L_OPT*0.5
    stretch_max = muscle.L_OPT*2.8
    N_stretch = 40
    muscle_stretch = np.arange(stretch_min, stretch_max, (stretch_max-stretch_min)/N_stretch)

    # Evalute for various muscle stimulation
    N_stim = 6
    muscle_stimulation = np.round(np.arange(N_stim)/(N_stim-1),2)
   
    # Set the initial condition
    x0 = [0.0, sys.muscle.L_OPT]
    # x0[0] --> muscle stimulation intial value
    # x0[1] --> muscle contracticle length initial value
    
    # Set the time for integration
    t_start = 0.0
    t_stop = 0.5
    time_step = 0.001
    time = np.arange(t_start, t_stop, time_step)
    
    activeF = np.zeros(N_stretch)
    passiveF = np.zeros(N_stretch)
    tendonF = np.zeros(N_stretch)
    lceF = np.zeros(N_stretch)
    
 # Plotting
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt.figure('Isometric Muscle Experiment') 
    
    for stim in range (len(muscle_stimulation)):
        pylog.info('Stimulation = {}'.format(stim))
        activeF = np.zeros(N_stretch)
        tendonF = np.zeros(N_stretch)
        index_acti_max = 0        
        for stretch in range (len(muscle_stretch)):
            # Run the integration
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   stimulation=muscle_stimulation[stim],
                                   muscle_length=muscle_stretch[stretch])
            activeF[stretch] = result.active_force[-1]/muscle.F_MAX
            if(activeF[stretch]>activeF[index_acti_max]):
                index_acti_max = stretch
            if(stim==0):
                passiveF[stretch] = result.passive_force[-1]/muscle.F_MAX
            tendonF[stretch] = result.tendon_force[-1]/muscle.F_MAX
            lceF[stretch] = result.l_ce[-1]
        
        color = colors[stim]
        #plt.plot(muscle_stretch*100/muscle.L_OPT, activeF, 'o' + color)
        #plt.plot(muscle_stretch*100/muscle.L_OPT, passiveF, '+' + color)
       
        plt.plot(lceF*100/muscle.L_OPT, activeF, color, label='Active Force - Stim = ' + str(round(muscle_stimulation[stim], 2)))

    plt.plot(lceF*100/muscle.L_OPT, passiveF, '+' + 'k', label='Passive Force')
    plt.title('Isometric Muscle Experiment')
    plt.xlabel('Contractile element length [% of $l_{opt}$]')
    plt.ylabel('Force [% of $F_{max}$]')
    plt.legend()
    plt.grid()
    
def exercise1c():


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
    
    # Evalute for various optimal length
    l0 = 0.11
    l_opt_list = l0*np.array([1,3])
    
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
            activeF[stretch] = result.active_force[-1]/muscle.F_MAX
            passiveF[stretch] = result.passive_force[-1]/muscle.F_MAX
            tendonF[stretch] = result.tendon_force[-1]/muscle.F_MAX
            lceF[stretch] = result.l_ce[-1]
                
        plt.subplot(n_subplot,n_subplot2,i+1)
        plt.plot(lceF, 100*activeF, label = 'Active')
        plt.plot(lceF, 100*passiveF, label = 'Passive')
        plt.plot(lceF, 100*tendonF, label = 'Tendon')
        plt.xlabel('Contractile element length')
        plt.ylabel('Force [% of F_max]')
        plt.ylim([0,200])
        plt.title('Optimal length : {}'.format(l_opt))
        plt.legend()
        plt.grid()
        plt.suptitle('Force-length curves for isometric muscle experiment with various muscle optimal length')
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
    
    # Subplots grid
    n_plot = len(muscle_stimulation)
    n_subplot = int(np.sqrt(n_plot-1)+1)
    if ((n_plot)<=n_subplot*(n_subplot-1)):
        fig, axes = plt.subplots(n_subplot,n_subplot-1)
        n_subplot2 = n_subplot-1
    else:
        fig, axes = plt.subplots(n_subplot,n_subplot)
        n_subplot2 = n_subplot

    for i,stim in enumerate(muscle_stimulation):
        max_velocity = np.zeros(N_load)
        activeF = np.zeros(N_load)
        passiveF = np.zeros(N_load)
        tendonF = np.zeros(N_load)
        for ind_load,load in enumerate(load_list):
            # Run the integration
            result = sys.integrate(x0=x0,
                                   time=time,
                                   time_step=time_step,
                                   time_stabilize=time_stabilize,
                                   stimulation=stim,
                                   load=load)
            if (result.l_mtc[-1] < (sys.muscle.L_OPT + sys.muscle.L_SLACK)):
                max_velocity[ind_load] = np.max(-result.v_ce)
            else:
                max_velocity[ind_load] = np.min(-result.v_ce)
            tendonF[ind_load] = result.tendon_force[-1]
            
        # Plotting
        plt.subplot(n_subplot,n_subplot2,i+1)
        plt.plot(max_velocity*100/-muscle.V_MAX, tendonF*100/muscle.F_MAX, 'k', label='Tendon force')
        plt.plot(max_velocity[max_velocity<=0]*100/-muscle.V_MAX, tendonF[max_velocity<=0]*100/muscle.F_MAX, 'b', label='lengthening')
        plt.plot(max_velocity[max_velocity>=-0]*100/-muscle.V_MAX, tendonF[max_velocity>=0]*100/muscle.F_MAX,'r',  label='shortening')
        plt.axvline(linewidth=1, linestyle='--', color='r')
        plt.xlabel('Velocity_max [% of V_MAX]')
        plt.ylabel('Tendon force [% of F_MAX]')
        plt.title('Stimulation : {}'.format(stim))
        plt.legend()
        plt.grid()
    
    plt.suptitle('Velocity-tension curves for isotonic muscle experiment with various muscle stimulations')
    fig.tight_layout()
                

def exercise1():

    #exercise1a()
    #exercise1b()
    #exercise1b_color()
    exercise1c()
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

