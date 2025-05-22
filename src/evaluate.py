#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This example simply simulates and visualizes the uncontrolled motion and
the model "falls down"."""

import os
import numpy as np
import sympy as sm
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pydy.codegen.ode_function_generators import generate_ode_function
from pygait2d import simulate

import derive

(mass_matrix, forcing_vector, _, constants, coordinates, speeds, specified,
     _, _, _, _) = derive.derive_equations_of_motion(gait_cycle_control=True)

constant_values = simulate.load_constants(
    constants, os.path.join(os.path.dirname(__file__), '..',
                            'data/example_constants.yml'))

print('Generating ODE function.')
rhs = generate_ode_function(
    forcing_vector,
    coordinates,
    speeds,
    constants=list(constant_values.keys()),
    mass_matrix=mass_matrix,
    specifieds=specified,
    generator='cython',
    constants_arg_type='array',
    # specifieds_arg_type='function',  # this did not work!! parameters ended up in t
)

# TODO: read a time series of belt speeds from file, and do a linear
# interpolation in the r_function

# define a function g(x,t) for all specified input signals r:
# question: can we code the controller here, rather than in the model?
#           we would also have to return dr/dx and the model would have to
#           use dr/dx
def r_function(x, t):
    r = np.zeros(len(specified))
    r[0] = 0.01*t;  # belt speed is the first specified input
    return r

# make an array of constant parameters
p = np.array(list(constant_values.values()))

time_vector = np.linspace(0.0, 50.0, num=5000)
initial_conditions = np.zeros(len(coordinates + speeds))
initial_conditions[1] = 1.0  # set hip above ground
initial_conditions[3] = np.deg2rad(5.0)  # right hip angle
initial_conditions[6] = -np.deg2rad(5.0)  # left hip angle
print('Simulating.')
trajectories = odeint(rhs, initial_conditions, time_vector, args=(r_function, p))

# plot the position of the trunk
print('Plotting.')
plt.plot(time_vector, trajectories[:,0], label='x')
plt.plot(time_vector, trajectories[:,1], label='y')
plt.xlabel('time (s)')
plt.ylabel('position (m)')
plt.legend()
plt.title('hip position')
plt.show()

