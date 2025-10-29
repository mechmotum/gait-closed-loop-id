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

# %%
# some settings
num_nodes = 2
h = 1.0  # time step does not matter, since it is a static equilibrium

# %%
# Derive the equations of motion
symbolics = derive_equations_of_motion(prevent_ground_penetration=False,
                                       treadmill=True, hand_of_god=False)

eom = symbolics.equations_of_motion
states = symbolics.states
num_states = len(states)

# %%
# The generalized coordinates are the hip lateral position :math:`q_{ax}` and
# veritcal position :math:`q_{ay}`, the trunk angle with respect to vertical
# :math:`q_a` and the relative joint angles:
#
# - right: hip (b), knee (c), ankle (d)
# - left: hip (e), knee (f), ankle (g)
#
# Each joint has a joint torque acting between the adjacent bodies.
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
bounds.update({k: (-0.1, 0.1)
               for k in [qa, qb, qc, qd, qe, qf, qg]})
# horizontal position and all speeds should be zero
bounds.update({k: (0.0, 0.0)
               for k in [qax, uax, uay, ua, ub, uc, ud, ue, uf, ug]})
# joint torques should be small
bounds.update({k: (-10.0, 10.0)
               for k in [Tb, Tc, Td, Te, Tf, Tg]})

# %%
# No instance constraints are needed
instance_constraints = None

# %%
# Objective is integral of sum of squared torques
num_angles = 6
torque_indices = num_states*num_nodes + np.arange(0, num_angles*num_nodes)


def obj(free):
    return np.sum(free[torque_indices]**2)


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[torque_indices] = 2.0*free[torque_indices]
    return grad


# %%
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
    instance_constraints=instance_constraints,
    bounds=bounds,
    time_symbol=time_symbol,
    tmp_dir='gait_codegen',  # enables binary caching
)

# %%
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
