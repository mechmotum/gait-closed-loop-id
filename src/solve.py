"""
Generates a half cycle of normal gait, by minimizing a combination of tracking
and error.

The resulting state trajectory will be used as a reference trajectory for a
known feedback controller to generate synthetic data for testing our controller
identification method.
"""
import os
from datetime import datetime
import logging

from opty import Problem
from pygait2d import simulate
from pygait2d.derive import derive_equations_of_motion
from pygait2d.segment import time_symbol
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm

from utils import (load_winter_data, load_sample_data, GAITDATAPATH, DATADIR,
                   plot_joint_comparison, generate_marker_equations, animate,
                   CALIBDATAPATH, body_segment_parameters_from_calibration,
                   plot_marker_comparison, SymbolDict, extract_values)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%H:%M:%S',
)

# Primary settings (capitalize)
EOM_SCALE = 10.0  # scaling factor for eom
GAIT_CYCLE_NUM = 45  # gait cycle to select from measurment data
GENFORCE_SCALE = 0.001  # convert to kN and kNm
MAKE_ANIMATION = True
NUM_NODES = 50  # number of time nodes for the half period
OBJ_WANGLTRACK = 100  # weight of mean squared angle tracking error (in rad)
OBJ_WMARKTRACK = 100  # weight of mean squared marker tracking error (in meters)
OBJ_WREG = 0.0  # weight of mean squared time derivatives (for regularization)
OBJ_WTORQUE = 100  # weight of the mean squared torque (in kNm) objective
STIFFNESS_EXP = 2  # exponent of the contact stiffness force
TRACK_ANGLES = True  # track joint angles
TRACK_MARKERS = True  # track markers

if not (TRACK_ANGLES and TRACK_MARKERS):
    raise ValueError('You must track joint angle or markers or both.')

# Load measurement data from Moore et al. 2015 if present, else load the
# normative Winter's data unless tracking markers is requested.
if os.path.exists(GAITDATAPATH):
    # load a gait cycle from our data (trial 20)
    (duration, walking_speed, num_angles, ang_data,
     marker_df, kinetic_df) = load_sample_data(
         NUM_NODES, gait_cycle_number=GAIT_CYCLE_NUM)
elif not TRACK_MARKERS:
    # load normal gait data from Winter's book
    duration, walking_speed, num_angles, ang_data = load_winter_data(NUM_NODES)
else:
    raise ValueError("Winter's data does not have markers to track.")

# Define the fixed time step in the simulation
h = duration/(NUM_NODES - 1)

# Derive the equations of motion
logger.info('Deriving the equations of motion.')
symbolics = derive_equations_of_motion(prevent_ground_penetration=False,
                                       treadmill=True, hand_of_god=False,
                                       stiffness_exp=STIFFNESS_EXP)
eom = symbolics.equations_of_motion
logger.info('Number of operations in eom: {}'.format(sm.count_ops(eom)))

# Do an overall scale, and then a unit conversion to kN and kNm
eom = EOM_SCALE * eom
for i in range(9):
    eom[9+i] = GENFORCE_SCALE * eom[9+i]

# Markers are in units meters, so no scaling applied
if TRACK_MARKERS:
    marker_coords, marker_eqs, marker_labels = generate_marker_equations(
        symbolics)
    eom = eom.col_join(sm.Matrix(marker_eqs))

# The generalized coordinates are the hip lateral position qax and veritcal
# position qay, the trunk angle with respect to vertical qa and the relative
# joint angles:
#
# - right: hip (b), knee (c), ankle (d)
# - left: hip (e), knee (f), ankle (g)
#
# Each joint has a joint torque acting between the adjacent bodies.
qax, qay, qa, qb, qc, qd, qe, qf, qg = symbolics.coordinates
uax, uay, ua, ub, uc, ud, ue, uf, ug = symbolics.speeds
Tb, Tc, Td, Te, Tf, Tg, v = symbolics.specifieds
angle_syms = [qb, qc, qd, qe, qf, qg]
torque_syms = [Tb, Tc, Td, Te, Tf, Tg]
num_states = len(symbolics.states)

# The constants are loaded from a file of realistic geometry, mass, inertia,
# and foot deformation properties of an adult human.
par_map = SymbolDict(simulate.load_constants(
    symbolics.constants, os.path.join(DATADIR, 'example_constants.yml')))
if STIFFNESS_EXP == 2:
    # Change stiffness value to give a 10mm static compression for a quadratic
    # force.
    par_map['kc'] = 1e7

# If there is calibration pose data, update the constants based on that
# subject's size.
if TRACK_MARKERS and os.path.exists(CALIBDATAPATH):
    # TODO : subject mass (70) should be loaded from meta data files.
    scaled_par = body_segment_parameters_from_calibration(CALIBDATAPATH, 70.0)
    for c in symbolics.constants:
        try:
            par_map[c] = scaled_par[c.name]
        except KeyError:
            pass

# Bound all the states to human realizable ranges.
#
# - The trunk should stay generally upright and be at a possible walking
#   height.
# - Only let the hip, knee, and ankle flex and extend to realistic limits.
# - Put a maximum on the peak torque values.
bounds = {
    qax: (-1.0, 1.0),
    qay: (0.5, 1.5),
    qa: np.deg2rad((-60.0, 60.0)),
    uax: (0.0, 10.0),
    uay: (-10.0, 10.0),
}
# hip
bounds.update({k: (-np.deg2rad(60.0), np.deg2rad(60.0))
               for k in [qb, qe]})
# knee
bounds.update({k: (-np.deg2rad(90.0), 0.0)
               for k in [qc, qf]})
# foot
bounds.update({k: (-np.deg2rad(40.0), np.deg2rad(40.0))
               for k in [qd, qg]})
# all rotational speeds
bounds.update({k: (-np.deg2rad(400.0), np.deg2rad(400.0))
               for k in [ua, ub, uc, ud, ue, uf, ug]})
# all joint torques
bounds.update({k: (-600.0, 600.0)
               for k in [Tb, Tc, Td, Te, Tf, Tg]})

# To enforce a half period, set the right leg's angles at the initial time to
# be equal to the left leg's angles at the final time and vice versa. The same
# goes for the joint angular rates.
instance_constraints = (
    qax.func(0*h) - qax.func(duration),
    qay.func(0*h) - qay.func(duration),
    qa.func(0*h) - qa.func(duration),
    qb.func(0*h) - qe.func(duration),
    qc.func(0*h) - qf.func(duration),
    qd.func(0*h) - qg.func(duration),
    qe.func(0*h) - qb.func(duration),
    qf.func(0*h) - qc.func(duration),
    qg.func(0*h) - qd.func(duration),
    uax.func(0*h) - uax.func(duration),
    uay.func(0*h) - uay.func(duration),
    ua.func(0*h) - ua.func(duration),
    ub.func(0*h) - ue.func(duration),
    uc.func(0*h) - uf.func(duration),
    ud.func(0*h) - ug.func(duration),
    ue.func(0*h) - ub.func(duration),
    uf.func(0*h) - uc.func(duration),
    ug.func(0*h) - ud.func(duration),
    # torques must also be periodic, because torques at t=0 are never used with
    # Backward Euler, and would otherwise become zero due to the cost function
    Tb.func(0*h) - Te.func(duration),
    Tc.func(0*h) - Tf.func(duration),
    Td.func(0*h) - Tg.func(duration),
    Te.func(0*h) - Tb.func(duration),
    Tf.func(0*h) - Tc.func(duration),
    Tg.func(0*h) - Td.func(duration),
)

if TRACK_MARKERS:
    for i, marker_sym in enumerate(marker_coords[:-2]):
        con = (marker_sym.func(0*h) - marker_coords[i + 2].func(duration),
               marker_coords[i + 2].func(0*h) - marker_sym.func(duration))
        instance_constraints += con

# The objective is a combination of squared torques and squared tracking errors
# Make indices for the free variables that are angles and torques.
# The final node is excluded, it is the first node of the next cycle
angle_indices = np.empty(num_angles*(NUM_NODES-1), dtype=np.int64)
num_torques = num_angles  # this is only tracking 6 angles, but there are 7 total angles
torque_indices = np.empty(num_torques*(NUM_NODES - 1), dtype=np.int64)
inodes = np.arange(0, NUM_NODES - 1)
for iangle in range(0, num_angles):
    # skip the first 3 DOFs and angles before iangle
    angle_indices[iangle*(NUM_NODES - 1) + inodes] = ((3 + iangle)*NUM_NODES +
                                                      inodes)
for itorque in range(0, num_torques):
    # skip the state trajectories, and torques before itorque
    torque_indices[itorque*(NUM_NODES-1) + inodes] = (
        (num_states+itorque)*NUM_NODES + inodes)

# make indices to all trajectory variables in the first N-1 nodes,
# for the regularization objective
reg_indices = np.empty((num_states+num_torques)*(NUM_NODES-1), dtype=np.int64)
for ivar in range(0, num_states + num_torques):
    # skip the variables before ivar
    reg_indices[ivar*(NUM_NODES-1) + inodes] = (ivar*NUM_NODES + inodes)


def obj(prob, free, obj_show=False):
    """
    Objective function::

        J =
          + WTOR*sum(joint_torque**2)
          + WANG*sum(joint_angle_error**2)
          + WMAR*sum(marker_error**2)
          + WREG*sum((dx/dt)**2)

    """
    # minimize mean joint torque
    tor_vals = extract_values(prob, free, *torque_syms, slice=(0, -1))
    f_tor = 1e-6*OBJ_WTORQUE*np.sum(tor_vals**2)/len(tor_vals)

    f_tot = f_tor

    # minimize mean angle tracking error
    ang_vals = extract_values(prob, free, *angle_syms, slice=(0, -1))
    f_track = OBJ_WANGLTRACK*np.sum((ang_vals - ang_data)**2)/len(ang_vals)

    f_tot += f_track

    # regularization cost is the mean of squared time derivatives of all
    # state variables (coordinates & speeds)
    reglead_vals = extract_values(
        prob, free, *(symbolics.states + torque_syms), slice=(1, None))
    reglag_vals = extract_values(
        prob, free, *(symbolics.states + torque_syms), slice=(None, -1))
    f_reg = OBJ_WREG*np.sum((reglead_vals -
                             reglag_vals)**2)/h**2/len(reglead_vals)

    f_tot += f_reg

    # minimize mean marker tracking error
    if TRACK_MARKERS:
        # vals -> shape(num_markers*(num_nodes - 1), 1)
        mar_vals = extract_values(prob, free, *marker_coords, slice=(None, -1))
        # TODO : Selecting the measured values can be outside of obj().
        meas_vals = marker_df[marker_labels].values.T.flatten()
        f_mar = OBJ_WMARKTRACK*np.sum((mar_vals - meas_vals)**2)/len(mar_vals)
        f_tot += f_mar

    if obj_show:
        msg = (f"   obj: {f_tot:.3f} = {f_tor:.3f}(torque) + "
               f"{f_track:.3f}(track) + {f_reg:.3f}(reg)")
        if TRACK_MARKERS:
            msg += f" + {f_mar:.3f}(marker)"
        print(msg)

    return f_tot


def obj_grad(prob, free):
    grad = np.zeros_like(free)
    grad[torque_indices] = (2e-6*OBJ_WTORQUE*free[torque_indices]/
                            torque_indices.size)
    grad[angle_indices] = (2.0*OBJ_WANGLTRACK*(free[angle_indices] - ang_data)/
                           angle_indices.size)
    grad[reg_indices] = grad[reg_indices] + (
        2.0*OBJ_WREG*(free[reg_indices+1]-free[reg_indices])/
        reg_indices.size/h**2)
    grad[reg_indices+1] = grad[reg_indices+1] + (
        2.0*OBJ_WREG*(free[reg_indices+1]-free[reg_indices])/
        reg_indices.size/h**2)
    # the regularization gradient could be coded more efficiently, but probably
    # not worth doing

    if TRACK_MARKERS:
        for var, lab in zip(marker_coords, marker_labels):
            vals = prob.extract_values(free, var)
            meas_vals = np.hstack((marker_df[lab].values,
                                   marker_df[lab].values[0]))
            prob.fill_free(grad, 2.0*OBJ_WMARKTRACK*(vals - meas_vals)/
                           len(vals)/len(marker_coords), var)

    return grad


# %%
# create the optimization problem
logger.info('Creating the opty problem.')

# Create a belt velocity signal v(t)
traj_map = {
    v: walking_speed*np.ones(NUM_NODES),
}

prob = Problem(
    obj,
    obj_grad,
    eom,
    symbolics.states,
    NUM_NODES,
    h,
    known_parameter_map=par_map,
    known_trajectory_map=traj_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
    time_symbol=time_symbol,
    tmp_dir='gait_codegen',  # enables binary caching
)
prob.add_option('max_iter', 3000)
prob.add_option('tol', 1e-3)
prob.add_option('constr_viol_tol', 1e-4)
prob.add_option('print_level', 0)




# %%
# make an initial guess from the standing solution
logger.info('Making an initial guess.')
fname = 'standing.csv'
if not os.path.exists(fname):
    logger.info('Solving standing problem.')
    # executes standing solution and generates 'standing.csv'
    import solve_standing  # this works, but probably not good python style
standing_sol = np.loadtxt(fname)
standing_state = standing_sol[0:num_states].reshape(-1, 1)  # coordinates and speeds as column vector
state_traj = np.tile(standing_state, (1, NUM_NODES))  # make NUM_NODES copies
tor_traj = np.zeros((num_angles, NUM_NODES))  # intialize torques to zero
initial_guess = np.concatenate((state_traj, tor_traj))  # complete trajectory
if TRACK_MARKERS:
    # TODO : The marker positions could be calculated from the generalized
    # coordinates.
    mar_traj = np.zeros((len(marker_coords), NUM_NODES))
    initial_guess = np.concatenate((initial_guess, mar_traj))
initial_guess = initial_guess.flatten()  # make a single row vector
np.random.seed(1)  # this makes the result reproducible
initial_guess = initial_guess + 0.01*np.random.random_sample(initial_guess.size)

# Solve the gait optimization problem for given belt speed
def solve_gait(speed, initial_guess=None):

    # change the belt speed signal
    traj_map[v] = speed*np.ones(NUM_NODES)

    # solve
    logger.info(datetime.now().strftime("%H:%M:%S") +
                f" solving for {speed:.3f} m/s")
    solution, info = prob.solve(initial_guess)
    # we accept solve_succeeded (0) and solved_to_acceptable_level (1)
    if info['status'] < 0:
        logger.info("IPOPT was not successful.")
        #breakpoint()

    # show the final objective function value and its contributions
    obj(prob, solution, obj_show=True)

    return solution


# solve for a series of increasing speeds, ending at the required speed
for speed in np.linspace(0.1, walking_speed, num=10):
    solution = solve_gait(speed, initial_guess)
    initial_guess = solution  # use this solution as guess for the next problem

# extract angles and torques
ang = solution[angle_indices].reshape(num_angles, NUM_NODES-1).transpose()
tor = solution[torque_indices].reshape(num_angles, NUM_NODES-1).transpose()
dat = ang_data.reshape(num_angles, NUM_NODES-1).transpose()

# construct a right side full gait cycle trajectory
ang = np.rad2deg(np.vstack((ang[:, 0:3], ang[:, 3:6], ang[1, 0:3])))
tor = np.vstack((tor[:, 0:3], tor[:, 3:6], tor[1, 0:3]))
dat = np.rad2deg(np.vstack((dat[:, 0:3], dat[:, 3:6], dat[1, 0:3])))
t = np.arange(2*NUM_NODES-1) * h

# use Winter's sign convention (knee flexion angle
# and hip/ankle extension torque)
ang[:, 1] = -ang[:, 1]
dat[:, 1] = -dat[:, 1]
tor[:, [0, 2]] = -tor[:, [0, 2]]

# Generate plots and animations
plot_joint_comparison(t, ang, tor, dat)

if TRACK_MARKERS:
    plot_marker_comparison(marker_coords, marker_labels, marker_df, prob,
                           solution)

plt.show()

if MAKE_ANIMATION:
    xs, rs, _ = prob.parse_free(solution)
    times = prob.time_vector(solution)
    animation = animate(symbolics, xs, rs, h, walking_speed, times, par_map,
                        STIFFNESS_EXP)
    animation.save('human_gait.gif', fps=int(1.0/h))
    plt.show()
