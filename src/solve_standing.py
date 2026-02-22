# solve_standing.py

# Finds a minimum-effort standing state.
# We do this as a trajectory optimization with two time points,
# so we can use opty.

# Import all necessary modules, functions, and classes:
import os
from opty import Problem
from pygait2d import simulate
from pygait2d.derive import derive_equations_of_motion
from pygait2d.segment import time_symbol
import numpy as np

num_nodes = 2
h = 1.0  # time step does not matter, since it is a static equilibrium

# Derive the equations of motion
symbolics = derive_equations_of_motion(prevent_ground_penetration=False,
                                       treadmill=True, hand_of_god=False)

eom = symbolics.equations_of_motion
states = symbolics.states
num_states = len(states)

qax, qay, qa, qb, qc, qd, qe, qf, qg = symbolics.coordinates
uax, uay, ua, ub, uc, ud, ue, uf, ug = symbolics.speeds
Tb, Tc, Td, Te, Tf, Tg, v = symbolics.specifieds

# %%
# The constants are loaded from a file of realistic geometry, mass, inertia,
# and foot deformation properties of an adult human.
par_map = simulate.load_constants(
    symbolics.constants, os.path.join(os.path.dirname(__file__), '..',
                                      'data/example_constants.yml'))

# %%
# Set belt velocity v(t) to zero
traj_map = {
    v: np.zeros(num_nodes),
}

# %%
# Bound all the states
bounds = {
    qay: (0.5, 1.5),
}
# trunk, hip, knee, ankle angles should all be small
bounds.update({k: (-0.1, 0.1) for k in [qa, qb, qc, qd, qe, qf, qg]})
# horizontal position and all speeds should be zero
bounds.update({k: (0.0, 0.0)
               for k in [qax, uax, uay, ua, ub, uc, ud, ue, uf, ug]})
# joint torques should be small
bounds.update({k: (-10.0, 10.0) for k in symbolics.joint_torques})


# Objective is integral of sum of squared torques
def obj(p, free):
    """Return sum of squares of joint torques."""
    tor_vals = p.extract_values(free, *symbolics.joint_torques)
    return np.sum(tor_vals**2)


def obj_grad(p, free):
    grad = np.zeros_like(free)
    tor_vals = p.extract_values(free, *symbolics.joint_torques)
    p.fill_free(grad, 2.0*tor_vals, *symbolics.joint_torques)
    return grad


# Create an optimization problem and solve it.
print("Creating the opty problem.")
prob = Problem(
    obj,
    obj_grad,
    eom,
    states,
    num_nodes,
    h,
    known_parameter_map=par_map,
    known_trajectory_map=traj_map,
    bounds=bounds,
    time_symbol=time_symbol,
)

# Find the optimal solution and save it if it converges.
#
initial_guess = (0.5*(prob.lower_bound + prob.upper_bound) +
                 0.01*(prob.upper_bound - prob.lower_bound)*
                 np.random.random_sample(prob.num_free))
solution, info = prob.solve(initial_guess)

# only keep the first node (the second is the same)
solution = solution[0:-1:2]
if info['status'] == 0:
    np.savetxt('standing.csv', solution, fmt='%.15f')
else:
    breakpoint()
