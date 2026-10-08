"""
Generates a half cycle of normal gait, by minimizing a combination of tracking
and error.

The resulting state trajectory will be used as a reference trajectory for a
known feedback controller to generate synthetic data for testing our controller
identification method.
"""
from datetime import datetime
import itertools
import logging
import os
import platform

from opty import Problem
from pygait2d import simulate
from pygait2d.derive import derive_equations_of_motion
from pygait2d.segment import time_symbol
import matplotlib.pyplot as plt
import numpy as np
import sympy as sm

from solve_standing import find_standing_state
from utils import (
    ANG_GAIT2D_COLS,
    CALIBDATAPATH,
    DATADIR,
    GRF_GAIT2D_COLS,
    SymbolDict,
    TOR_GAIT2D_COLS,
    animate,
    body_segment_parameters_from_calibration,
    extract_values,
    extract_values_diff,
    fill_free,
    full_gait_from_half,
    generate_grf_equations,
    generate_marker_equations,
    generate_planar_grf_func,
    load_sample_data,
    load_winter_data_frame,
    plot_joint_comparison,
    plot_marker_comparison,
    tile_standing,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%H:%M:%S',
)

# Primary settings (capitalize)
EOM_SCALE = 10.0  # scaling factor for eom
GAIT_CYCLE_NUM = 145  # gait cycle to select from measurment data
GENFORCE_SCALE = 0.001  # convert to kN and kNm
LINEAR_SOLVER = 'ma57'  # passed to IPOPT mumps, spral, ma57, ma77, ma86, ma97
MAKE_ANIMATION = True
NUM_NODES = 50  # number of time nodes for the half period
SEED = True  # set to integer value for specific seed value, True(=1), or False
STIFFNESS_EXP = 2  # exponent of the contact stiffness force
SUBJECT_MASS = 70.0  # kg of subject from trial 20, TODO: extract from metadata
USE_WINTER_DATA = True  # if we want to track Winter's gait data
# Remove parts of the objective by setting to integer 0.
WANG = 1000.0  # weight of mean squared angle tracking error (in rad)
WGRF = 0.001  # weight of mean squared GRF tracking error (in Newtons)
WMAR = 0  # weight of mean squared marker tracking error (in meters)
WREG = 1e-6  # weight of mean squared time derivatives
WTOR = 1000.0  # weight of the mean squared torque (in kNm) objective

# Load half cycle [0%, 50%] measurement data from Moore et al. 2015 (trial 20)
# or normative Winter's data unless tracking markers is requested.
if USE_WINTER_DATA:
    if WMAR != 0:
        raise ValueError("Winter's data does not have markers to track.")
    df = load_winter_data_frame(num_nodes=NUM_NODES, half_cycle=True)
else:
    df = load_sample_data(NUM_NODES, gait_cycle_number=GAIT_CYCLE_NUM)

duration = df['Time'].values[-1]  # time @ 50%
h = duration/(NUM_NODES - 1)  # fixed time step in the simulation
walking_speed = df['Speed'].mean()

# Derive the equations of motion
logger.info('Deriving the equations of motion.')
syms = derive_equations_of_motion(
    prevent_ground_penetration=False,
    treadmill=True,
    hand_of_god=False,
    stiffness_exp=STIFFNESS_EXP,
)
eom = syms.equations_of_motion
logger.info('Number of operations in eom: {}'.format(sm.count_ops(eom)))

# Do an overall scale, and then a unit conversion to kN and kNm
eom = EOM_SCALE*eom
for i in range(9):
    eom[9+i] = GENFORCE_SCALE*eom[9+i]

# Extract angle measurement data [0%, 50%) and flatten
ang_meas = df[ANG_GAIT2D_COLS].values[:-1, :].transpose().flatten()

# Markers are in units meters, so no scaling applied
if WMAR != 0:
    marker_syms, marker_eqs, marker_labels = generate_marker_equations(syms)
    eom = eom.col_join(sm.Matrix(marker_eqs))
    mar_meas = df[marker_labels].values[:-1, :].T.flatten()

# Ground reaction forces are in units Newtons
if WGRF != 0:
    grf_syms, grf_eqs, grf_labels = generate_grf_equations(syms)
    eom = eom.col_join(grf_eqs)
    grf_meas = df[grf_labels].values[:-1, :].T.flatten()

# The generalized coordinates are the hip lateral position qax and veritcal
# position qay, the trunk angle with respect to vertical qa and the relative
# joint angles:
#
# - right: hip (b), knee (c), ankle (d)
# - left: hip (e), knee (f), ankle (g)
#
# Each joint has a joint torque acting between the adjacent bodies.
qax, qay, qa, qb, qc, qd, qe, qf, qg = syms.coordinates
uax, uay, ua, ub, uc, ud, ue, uf, ug = syms.speeds
Tb, Tc, Td, Te, Tf, Tg, v = syms.specifieds
reg_syms = syms.states + syms.joint_torques

# The constants are loaded from a file of realistic geometry, mass, inertia,
# and foot deformation properties of an adult human.
par_map = SymbolDict(simulate.load_constants(
    syms.constants, os.path.join(DATADIR, 'example_constants.yml')))
if STIFFNESS_EXP == 2:
    # Change stiffness value to give a 10mm static compression for a 1 kN load
    # with a quadratic force model.
    par_map['kc'] = 1e7

# If there is calibration pose data, update the constants based on that
# subject's size.
if (not USE_WINTER_DATA) and os.path.exists(CALIBDATAPATH):
    scaled_par = body_segment_parameters_from_calibration(CALIBDATAPATH,
                                                          SUBJECT_MASS)
    # TODO : get this to work with SymbolDict: par_map.update(scaled_par)
    for c in syms.constants:
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
    uax: (-1.0, 1.0),
    uay: (-1.0, 1.0),
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
# TODO : Add bounds for marker trajectories and ground reaction forces.

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

# When tracking markers, simulated marker trajectories must be periodic too
if WMAR != 0:
    for (lx, ly, rx, ry) in itertools.zip_longest(*[iter(marker_syms)]*4):
        # group per marker: (ank_lx(t), ank_ly(t), ank_rx(t), ank_ry(t))
        con = (
            lx.func(0*h) - rx.func(duration),
            rx.func(0*h) - lx.func(duration),
            ly.func(0*h) - ry.func(duration),
            ry.func(0*h) - ly.func(duration),
        )
        instance_constraints += con

# When tracking ground reaction forces, simulation must be periodic too
if WGRF != 0:
    Frx, Fry, Flx, Fly = grf_syms
    con = (
        Flx.func(0*h) - Frx.func(duration),
        Frx.func(0*h) - Flx.func(duration),
        Fly.func(0*h) - Fry.func(duration),
        Fry.func(0*h) - Fly.func(duration),
    )
    instance_constraints += con


def obj(prob, free, obj_show=False):
    """
    Objective function::

        J = WTOR*mean(joint_torque**2)
          + WANG*mean(joint_angle_error**2)
          + WREG*mean((dx/dt)**2)
          + WMAR*mean(marker_error**2)
          + WGRF*mean(grf_error**2)

    The final node is excluded from all means. Due to symmetry and
    periodicity constraints, it is the mirror image of the first node,
    and we don't want to include it twice.

    """
    # NOTE : slice(0, -1) is used to avoid double counting the periodic
    # duplicate values

    # minimize mean joint torque
    tor_vals = extract_values(prob, free, *syms.joint_torques, slice=(0, -1))
    f_tor = 1e-6*WTOR*np.sum(tor_vals**2)/len(tor_vals)

    f_tot = f_tor

    # minimize mean angle tracking error
    if WANG != 0:
        ang_vals = extract_values(prob, free, *syms.joint_angles,
                                  slice=(0, -1))
        f_ang = WANG*np.sum((ang_vals - ang_meas)**2)/len(ang_vals)
        f_tot += f_ang

    # smooth all regularization trajectories
    if WREG != 0:
        diff = extract_values_diff(prob, free, *reg_syms)
        f_reg = WREG*np.sum(diff**2)/h**2/len(diff)
        f_tot += f_reg

    # minimize mean marker tracking error
    if WMAR != 0:
        # vals -> shape(num_markers*(num_nodes - 1), 1)
        mar_vals = extract_values(prob, free, *marker_syms, slice=(0, -1))
        f_mar = WMAR*np.sum((mar_vals - mar_meas)**2)/len(mar_vals)
        f_tot += f_mar

    # minimize mean ground reaction force tracking error
    if WGRF != 0:
        grf_vals = extract_values(prob, free, *grf_syms, slice=(0, -1))
        f_grf = WGRF*np.sum((grf_vals - grf_meas)**2)/len(grf_vals)
        f_tot += f_grf

    if obj_show:
        msg = (f"   obj: {f_tot:.3f} = {f_tor:.3f}(torque)")
        if WREG != 0:
            msg += f" + {f_reg:.3f}(reg)"
        if WANG != 0:
            msg += f" + {f_ang:.3f}(angle)"
        if WMAR != 0:
            msg += f" + {f_mar:.3f}(marker)"
        if WGRF != 0:
            msg += f" + {f_grf:.3f}(grf)"
        print(msg)

    return f_tot


def obj_grad(prob, free):

    grad = np.zeros_like(free)

    tor_vals = extract_values(prob, free, *syms.joint_torques, slice=(0, -1))
    fill_free(prob, grad, 2e-6*WTOR*tor_vals/len(tor_vals),
              *syms.joint_torques, slice=(0, -1))

    if WANG != 0:
        ang_vals = extract_values(prob, free, *syms.joint_angles,
                                  slice=(0, -1))
        fill_free(prob, grad,
                  2.0*WANG*(ang_vals - ang_meas)/len(ang_vals),
                  *syms.joint_angles, slice=(0, -1))
    if WREG != 0:
        # NOTE : The regularization should be added on top of the tor_vals and
        # ang_vals.
        diff = extract_values_diff(prob, free, *reg_syms)
        reg_grad = WREG*2.0*diff/h**2/len(diff)
        # NOTE : Add twice with a shift for correct gradient.
        fill_free(prob, grad, -reg_grad, *reg_syms, slice=(None, -1), add=True)
        fill_free(prob, grad, reg_grad, *reg_syms, slice=(1, None), add=True)

    if WMAR != 0:
        mar_vals = extract_values(prob, free, *marker_syms, slice=(0, -1))
        fill_free(prob, grad,
                  2.0*WMAR*(mar_vals - mar_meas)/len(mar_vals),
                  *marker_syms, slice=(0, -1))

    if WGRF != 0:
        grf_vals = extract_values(prob, free, *grf_syms, slice=(0, -1))
        fill_free(prob, grad,
                  2.0*WGRF*(grf_vals - grf_meas)/len(grf_vals),
                  *grf_syms, slice=(0, -1))

    return grad


# Create a belt velocity signal v(t)
traj_map = {
    v: df['Speed'].values,  # shape(NUM_NODES,)
}

logger.info('Creating the opty problem.')
prob = Problem(
    obj,
    obj_grad,
    eom,
    syms.states,
    NUM_NODES,
    h,
    known_parameter_map=par_map,
    known_trajectory_map=traj_map,
    instance_constraints=instance_constraints,
    bounds=bounds,
    time_symbol=time_symbol,
    tmp_dir='gait_codegen',  # enables binary caching
)

# When not tracking markers, the dynamics and cost function are invariant
# to a constant horizontal translation of the trajectory, so there is no
# unique solution. We therefore require qax(t=0) = 0
# (it would be better to use the bounds to require free[0]=zero, but I don't
# know how to do this in opty, I only see bounds on the whole trajectory)
if WMAR == 0:
    # instance_constraints += (qax.func(0*h),)
    prob.lower_bound[0] = 0.0
    prob.upper_bound[0] = 0.0

if LINEAR_SOLVER.startswith('ma'):
    # TODO : Load correct shared lib on Windows/Darwin
    if platform.system() == 'Linux':
        prob.add_option('hsllib', 'libcoinhsl.so')
    prob.add_option('linear_solver', LINEAR_SOLVER)
    if LINEAR_SOLVER == 'ma57':
        prob.add_option('ma57_pivot_order', 2)

prob.add_option('max_iter', 3000)
prob.add_option('tol', 1e-3)
prob.add_option('constr_viol_tol', 1e-4)
prob.add_option('print_level', 0)
#prob.add_option('derivative_test', 'first-order')

# make an initial guess from the standing solution
logger.info('Making an initial guess.')
fname = os.path.join(DATADIR, 'standing.csv')
if not os.path.exists(fname):
    logger.info('Solving standing problem.')
    standing_sol = find_standing_state()
else:
    standing_sol = np.loadtxt(fname)

initial_guess = tile_standing(standing_sol, NUM_NODES, len(syms.joint_angles),
                              len(syms.states))
if WMAR != 0:
    # TODO : The marker positions could be calculated from the generalized
    # coordinates.
    mar_traj = np.zeros((len(marker_syms), NUM_NODES))
    initial_guess = np.concatenate((initial_guess, mar_traj))
if WGRF != 0:
    grf_traj = np.zeros((len(grf_syms), NUM_NODES))
    initial_guess = np.concatenate((initial_guess, grf_traj))
initial_guess = initial_guess.flatten()  # make a single row vector
if SEED:
    np.random.seed(SEED)  # this makes the result reproducible
initial_guess = (initial_guess +
                 0.01*np.random.random_sample(len(initial_guess)))


# Solve the gait optimization problem for given belt speed
def solve_gait(speed, initial_guess):

    # change the belt speed signal
    traj_map[v] = speed*np.ones(NUM_NODES)

    # solve
    logger.info(datetime.now().strftime("%H:%M:%S") +
                f" solving for {speed:.3f} m/s")
    solution, info = prob.solve(initial_guess)
    # we accept solve_succeeded (0) and solved_to_acceptable_level (1)
    if info['status'] < 0:
        logger.info("IPOPT was not successful.")

    # show the final objective function value and its contributions
    obj(prob, solution, obj_show=True)

    return solution


# solve for a series of increasing speeds, ending at the required speed
for speed in np.linspace(0.1, walking_speed, num=10):
    solution = solve_gait(speed, initial_guess)
    initial_guess = solution  # use this solution as guess for the next problem

# TODO : Move data preparation for plots into functions in utils.py
# extract angles and torques
ang_sol = extract_values(prob, solution, *syms.joint_angles,
                         slice=(0, -1)).reshape(len(syms.joint_angles),
                                                NUM_NODES-1).transpose()
tor_sol = extract_values(prob, solution, *syms.joint_torques,
                     slice=(0, -1)).reshape(len(syms.joint_torques),
                                            NUM_NODES-1).transpose()
if WGRF != 0:
    # TODO : Extract the GRFs from the Winter's data also.
    # Frx(t), Fry(t), Flx(t), Fly(t)
    # N-1 x 4
    grf_sol = extract_values(prob, solution, *grf_syms,
                             slice=(0, -1)).reshape(len(grf_syms),
                                                    NUM_NODES-1).transpose()
else:
    eval_grf = generate_planar_grf_func(syms)
    xs, rs, _ = prob.parse_free(solution)
    grf_sol = eval_grf(
        xs,
        np.vstack((
            rs[:6, :],  # r, shape(q, N)
            traj_map[v])  # belt speed shape(1, N)
        ),
        np.repeat(np.atleast_2d(np.array(list(par_map.values()))).T,
                  xs.shape[0], axis=1)  # p, shape(r, N)
    )  # shape(N, 4)
    grf_sol = grf_sol[:-1, :]  # shape(N-1, 4)

ang_meas = df[ANG_GAIT2D_COLS].values[:-1, :]
tor_meas = df[TOR_GAIT2D_COLS].values[:-1, :]
grf_meas = df[GRF_GAIT2D_COLS].values[:-1, :]

# construct a right side full gait cycle trajectory
plot_time = np.linspace(0.0, duration - h, num=2*NUM_NODES - 1)

ang_sol = full_gait_from_half(ang_sol)
tor_sol = full_gait_from_half(tor_sol)
grf_sol = full_gait_from_half(grf_sol)

ang_meas = full_gait_from_half(ang_meas)
tor_meas = full_gait_from_half(tor_meas)
grf_meas = full_gait_from_half(grf_meas)

# Generate plots and animations
plot_joint_comparison(plot_time, ang_sol, tor_sol, ang_meas,
                      torques_meas=tor_meas, grf=grf_sol, grf_meas=grf_meas)

if WMAR != 0:
    plot_marker_comparison(marker_syms, marker_labels, df, prob, solution)

plt.show()

if MAKE_ANIMATION:
    xs, rs, _ = prob.parse_free(solution)
    times = prob.time_vector(solution)
    animation = animate(syms, xs, rs, h, walking_speed, times, par_map,
                        STIFFNESS_EXP, plot_time, grf_meas)
    animation.save('human_gait.gif', fps=int(1.0/h))
    plt.show()
