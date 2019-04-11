""" Lab 6 Exercise 2

This file implements the pendulum system with two muscles attached

"""

from math import sqrt

import cmc_pylog as pylog
import numpy as np
from matplotlib import pyplot as plt

from cmcpack import DEFAULT
from cmcpack.plot import save_figure
from muscle import Muscle
from muscle_system import MuscleSytem
from neural_system import NeuralSystem
from pendulum_system import PendulumSystem
from system import System
from system_animation import SystemAnimation
from system_parameters import (MuscleParameters, NetworkParameters,
                               PendulumParameters)
from system_simulation import SystemSimulation


# Global settings for plotting
# You may change as per your requirement
plt.rc('lines', linewidth=2.0)
plt.rc('font', size=12.0)
plt.rc('axes', titlesize=14.0)     # fontsize of the axes title
plt.rc('axes', labelsize=14.0)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=14.0)    # fontsize of the tick labels
plt.rc('ytick', labelsize=14.0)    # fontsize of the tick labels

DEFAULT["save_figures"] = True

def exercise2():
    """ Main function to run for Exercise 2.
    """

    # Define and Setup your pendulum model here
    # Check PendulumSystem.py for more details on Pendulum class
    pendulum_params = PendulumParameters()  # Instantiate pendulum parameters
    pendulum_params.L = 0.5  # To change the default length of the pendulum
    pendulum_params.m = 1.  # To change the default mass of the pendulum
    pendulum = PendulumSystem(pendulum_params)  # Instantiate Pendulum object

    #### CHECK OUT PendulumSystem.py to ADD PERTURBATIONS TO THE MODEL #####

    pylog.info('Pendulum model initialized \n {}'.format(
        pendulum.parameters.showParameters()))

    # Define and Setup your pendulum model here
    # Check MuscleSytem.py for more details on MuscleSytem class
    M1_param = MuscleParameters()  # Instantiate Muscle 1 parameters
    M1_param.f_max = 1500  # To change Muscle 1 max force
    M2_param = MuscleParameters()  # Instantiate Muscle 2 parameters
    M2_param.f_max = 1500  # To change Muscle 2 max force
    M1 = Muscle(M1_param)  # Instantiate Muscle 1 object
    M2 = Muscle(M2_param)  # Instantiate Muscle 2 object
    # Use the MuscleSystem Class to define your muscles in the system
    muscles = MuscleSytem(M1, M2)  # Instantiate Muscle System with two muscles
    pylog.info('Muscle system initialized \n {} \n {}'.format(
        M1.parameters.showParameters(),
        M2.parameters.showParameters()))

    # 2a : set of muscle 1 attachment points
    
    m1_origin = np.array([[-0.17, 0.0]])  # Origin of Muscle 1
    m1_insertion = np.array([[0.0, -0.17], [0.0, -0.3], [0.0, -0.4], [0.0, -0.5]])  # Insertion of Muscle 1
    
    theta = np.linspace(-np.pi/2,np.pi/2)
        
    m_lengths = np.zeros((len(m1_insertion),len(theta)))
    m_moment_arms = np.zeros((len(m1_insertion),len(theta)))
    leg=[]
    for i in range(0,len(m1_insertion)):
        m_lengths[i,:]=np.sqrt(m1_origin[0,0]**2 + m1_insertion[i,1]**2 +
                         2 * np.abs(m1_origin[0,0]) * np.abs(m1_insertion[i,1]) * np.sin(theta))
        m_moment_arms[i,:]= m1_origin[0,0] * m1_insertion[i,1] * np.cos(theta) / m_lengths[i,:]
        leg.append('Origin: {}m, Insertion: {}m'.format(m1_origin[0,0],m1_insertion[i,1]))
        
    # Plotting
    plt.figure('2a length')
    plt.title('Length of M1 with respect to the position of the limb') 
    for i in range(0,len(m_lengths)):
        plt.plot(theta*180/np.pi, m_lengths[i,:])
    plt.plot((theta[0]*180/np.pi,theta[len(theta)-1]*180/np.pi),(0.11,0.11), ls='dashed')
    leg.append('l_opt')
    plt.plot((theta[0]*180/np.pi,theta[len(theta)-1]*180/np.pi),(0.13,0.13), ls='dashed')
    leg.append('l_slack') 
    plt.xlabel('Position [deg]')
    plt.ylabel('Muscle length [m]')
    plt.legend(leg)
    plt.grid()
    
    plt.figure('2a moment')
    plt.title('Moment arm over M1 with respect to the position of the limb')
    for i in range(0,len(m_moment_arms)):
        plt.plot(theta*180/np.pi, m_moment_arms[i,:])
    plt.xlabel('Position [deg]')
    plt.ylabel('Moment arm [m]')
    plt.legend(leg)
    plt.grid()
    plt.savefig('2_a.png')
    
    
    # 2b : simple activation wave forms
        
    # Muscle 2 attachement point
    m2_origin = np.array([0.17, 0.0])  # Origin of Muscle 2
    m2_insertion = np.array([0.0, -0.17])  # Insertion of Muscle 2
    
    # Attach the muscles
    muscles.attach(np.array([m1_origin[0,:], m1_insertion[0,:]]),
                   np.array([m2_origin, m2_insertion]))

    # Create a system with Pendulum and Muscles using the System Class
    # Check System.py for more details on System class
    sys = System()  # Instantiate a new system
    sys.add_pendulum_system(pendulum)  # Add the pendulum model to the system
    sys.add_muscle_system(muscles)  # Add the muscle model to the system

    ##### Time #####
    t_max = 2.5  # Maximum simulation time
    time = np.arange(0., t_max, 0.001)  # Time vector

    ##### Model Initial Conditions #####
    x0_P = np.array([np.pi/4, 0.])  # Pendulum initial condition

    # Muscle Model initial condition
    x0_M = np.array([0., M1.L_OPT, 0., M2.L_OPT])

    x0 = np.concatenate((x0_P, x0_M))  # System initial conditions

    ##### System Simulation #####
    # For more details on System Simulation check SystemSimulation.py
    # SystemSimulation is used to initialize the system and integrate
    # over time

    sim = SystemSimulation(sys)  # Instantiate Simulation object

    # Add muscle activations to the simulation
    # Here you can define your muscle activation vectors
    # that are time dependent

    sin_frequency = 2 #Hz
    amp_stim = 1
    phase_shift = np.pi
    act1 = np.zeros((len(time),1))
    act2 = np.zeros((len(time),1))
    for i in range(0,len(time)):
        act1[i,0] = amp_stim*(1+np.sin(2*np.pi*sin_frequency*time[i]))/2
        act2[i,0] = amp_stim*(1+ np.sin(2*np.pi*sin_frequency*time[i] + phase_shift))/2
    
    plt.figure('2b activation')
    plt.plot(time,act1)
    plt.plot(time,act2)
    plt.legend(["Activation for muscle 1", "Activation for muscle 2"])
    plt.title('Activation for muscle 1 and 2 with simple activation wave forms')
    plt.xlabel("Time [s]")
    plt.ylabel("Activation")
    plt.savefig('2_b_activation.png')
    plt.show()

    activations = np.hstack((act1, act2))

    # Method to add the muscle activations to the simulation

    sim.add_muscle_activations(activations)

    # Simulate the system for given time

    sim.initalize_system(x0, time)  # Initialize the system state
    
    #: If you would like to perturb the pedulum model then you could do
    # so by
    sim.sys.pendulum_sys.parameters.PERTURBATION = True
    # The above line sets the state of the pendulum model to zeros between
    # time interval 1.2 < t < 1.25. You can change this and the type of
    # perturbation in
    # pendulum_system.py::pendulum_system function
    
    # Integrate the system for the above initialized state and time
    sim.simulate()
    
    # Obtain the states of the system after integration
    # res is np.array [time, states]
    # states vector is in the same order as x0
    res = sim.results()
    
    # Plotting the results
    plt.figure('2b phase')
    plt.title('Pendulum Phase')
    plt.plot(res[:, 1], res[:, 2])
    plt.xlabel('Position [rad]')
    plt.ylabel('Velocity [rad/s]')
    plt.grid()
    plt.savefig('2_b_phase.png')
    plt.show()
    
    plt.figure('2b oscillations')
    plt.title('Pendulum Oscillations')
    plt.plot(time,res[:, 1])
    plt.xlabel('Time [s]')
    plt.ylabel('Position [rad]')
    plt.grid()
    plt.savefig('2_b_oscillations.png')
    plt.show()
    
    # To animate the model, use the SystemAnimation class
    # Pass the res(states) and systems you wish to animate
    simulation = SystemAnimation(res, pendulum, muscles)
    # To start the animation
    if DEFAULT["save_figures"] is False:
        simulation.animate()

    if not DEFAULT["save_figures"]:
        plt.show()
    else:
        figures = plt.get_figlabels()
        pylog.debug("Saving figures:\n{}".format(figures))
        for fig in figures:
            plt.figure(fig)
            save_figure(fig)
            plt.close(fig)
    
 # 2c : relationship between stimulation frequency and amplitude
    
    # Effect of frequency
    stim_frequency_range = np.array([0.05,0.1,0.5,1,5,10,50,100,500]) #Hz
    stim_amp = 1
    phase_shift = np.pi
    frequency_pend=np.zeros(len(stim_frequency_range))
    amplitude_pend=np.zeros(len(stim_frequency_range))
    
    for j,stim_frequency in enumerate(stim_frequency_range):
        period = 1/stim_frequency
        t_max = 10*period  # Maximum simulation time
        time = np.arange(0., t_max, 0.001*period)  # Time vector

        act1 = np.zeros((len(time),1))
        act2 = np.zeros((len(time),1))
        act1[:,0] = stim_amp*(1 + np.sin(2*np.pi*stim_frequency*time))/2
        act2[:,0] = stim_amp*(1+ np.sin(2*np.pi*stim_frequency*time + phase_shift))/2
        activations = np.hstack((act1, act2))
        sim.add_muscle_activations(activations)
        sim.initalize_system(x0, time)  # Initialize the system state
        sim.simulate()
        res = sim.results()  
        # computing the frequency and amplitude
        angular_position = res[:,1]
        signal_stat = angular_position[int(len(angular_position)/2):len(angular_position)]
        index_zeros = np.where(np.diff(np.sign(signal_stat)))[0]
        deltas = np.diff(index_zeros)
        delta = np.mean(deltas)
        period = 2*delta*0.001*period
        frequency_pend[j] = 1/period
        amplitude_pend[j] = (np.max(signal_stat)-np.min(signal_stat))/2

    # Plotting
    plt.figure('2c : effect of frequency')
    plt.subplot(121)
    plt.loglog(stim_frequency_range,frequency_pend)
    plt.grid()
    plt.xlabel('Stimulation Frequency in Hz')
    plt.ylabel('Pendulum Oscillation Frequency [Hz]')
    plt.subplot(122)
    plt.loglog(stim_frequency_range,amplitude_pend)
    plt.grid()
    plt.xlabel('Stimulation Frequency in Hz')
    plt.ylabel('Pendulum Oscillation Amplitude [rad]')
    plt.save('2c_frequency')
    plt.show()

    # Effect of amplitude
    stim_frequency = 10 #Hz
    stim_amp_range = np.arange(0,1.1,0.1)
    phase_shift = np.pi
    frequency_pend=np.zeros(len(stim_amp_range))
    amplitude_pend=np.zeros(len(stim_amp_range))
    
    for j,stim_amp in enumerate(stim_amp_range):
        period = 1/stim_frequency
        t_max = 5*period  # Maximum simulation time
        time = np.arange(0., t_max, 0.001*period)  # Time vector

        act1 = np.zeros((len(time),1))
        act2 = np.zeros((len(time),1))
        act1[:,0] = stim_amp*(1 + np.sin(2*np.pi*stim_frequency*time))/2
        act2[:,0] = stim_amp*(1+ np.sin(2*np.pi*stim_frequency*time + phase_shift))/2
        activations = np.hstack((act1, act2))
        sim.add_muscle_activations(activations)
        sim.initalize_system(x0, time)  # Initialize the system state
        sim.simulate()
        res = sim.results()  
        # computing the frequency and amplitude
        angular_position = res[:,1]
        signal_stat = angular_position[int(len(angular_position)/2):len(angular_position)]
        index_zeros = np.where(np.diff(np.sign(signal_stat)))[0]
        deltas = np.diff(index_zeros)
        delta = np.mean(deltas)
        period = 2*delta*0.001*period
        frequency_pend[j] = 1/period
        amplitude_pend[j] = (np.max(signal_stat)-np.min(signal_stat))/2
        
    frequency_pend[0] = 0.0;
    
    # Plotting
    plt.figure('2c : effect of amplitude')
    plt.subplot(121)
    plt.plot(stim_amp_range,frequency_pend)
    plt.grid()
    plt.xlabel('Stimulation Amplitude')
    plt.ylabel('Pendulum Oscillation Frequency [Hz]')
    plt.subplot(122)
    plt.plot(stim_amp_range,amplitude_pend)
    plt.grid()
    plt.xlabel('Stimulation Amplitude')
    plt.ylabel('Pendulum Oscillation Amplitude [rad]')
    plt.save('2c_amplitude')
    plt.show()
    
    
if __name__ == '__main__':
    from cmcpack import parse_args
    parse_args()
    exercise2()

