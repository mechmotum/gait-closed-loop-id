# solve.py

# Generates a half cycle of normal gait, by minimizing a combination
# of tracking and error.

# The resulting state trajectory will be used as a reference trajectory
# for a known feedback controller to generate synthetic data for
# testing our controller identification method.

# Import all necessary modules, functions, and classes:
import os
from datetime import datetime
import logging

from opty import Problem
from pygait2d import simulate
from pygait2d.derive import derive_equations_of_motion
from pygait2d.segment import time_symbol, contact_force, time_varying
from symmeplot.matplotlib import Scene3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sympy as sm

from utils import (load_winter_data, load_sample_data, DATAPATH,
                   plot_joint_comparison)

root = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%H:%M:%S',
)

# %%
# some settings
make_animation = True
num_nodes = 50       # number of time nodes for the half period
genforce_scale = 0.001  # convert to kN and kNm
eom_scale = 10.0     # scaling factor for eom
# genforce_scale = 1 # convert to kN and kNm
# eom_scale = 1     # scaling factor for eom
obj_Wtorque = 100   # weight of torque objective
obj_Wtrack = 100      # weight of tracking objective
TRACK_MARKERS = True
GAIT_CYCLE_NUM = 45

if os.path.exists(DATAPATH):
    # load a gait cycle from our data (trial 20)
    (duration, walking_speed, num_angles, ang_data,
     marker_df) = load_sample_data(num_nodes, gait_cycle_number=GAIT_CYCLE_NUM)
elif not TRACK_MARKERS:
    # load normal gait data from Winter's book
    duration, walking_speed, num_angles, ang_data = load_winter_data(num_nodes)
else:
    raise ValueError("Winter's data does not have markers to track.")

h = duration/(num_nodes - 1)

# %%
# Derive the equations of motion
logging.info('Deriving the equations of motion.')
symbolics = derive_equations_of_motion(prevent_ground_penetration=False,
                                       treadmill=True, hand_of_god=False)
eom = symbolics.equations_of_motion


def generate_marker_equations(symbolics):

    O, N = symbolics.origin, symbolics.inertial_frame
    trunk, rthigh, rshank, rfoot, lthigh, lshank, lfoot = symbolics.segments

    points = {
        'ank_l': lshank.joint,  # left ankle
        'ank_r': rshank.joint,  # right ankle
        #'hel_l': lfoot.heel,
        #'hel_r': rfoot.heel,
        #'hip_m': trunk.joint,  # hip
        #'kne_l': lthigh.joint,  # left knee
        #'kne_r': rthigh.joint,  # right knee
        #'toe_l': lfoot.toe,
        #'toe_r': rfoot.toe,
        #'tor_m': trunk.mass_center,
    }

    variables = []
    equations = []

    for lab, point in points.items():
        x, y = time_varying(f'{lab}x, {lab}y')
        variables += [x, y]
        # TODO : Manage belt speed if that matters.
        x_eq = x - point.pos_from(O).dot(N.x)
        y_eq = y - point.pos_from(O).dot(N.y)
        equations += [x_eq, y_eq]

    return variables, equations


# do an overall scale, and then a unit conversion to kN and kNm
eom = eom_scale * eom
for i in range(9):
    eom[9+i] = genforce_scale * eom[9+i]

if TRACK_MARKERS:
    marker_coords, marker_eqs = generate_marker_equations(symbolics)
    ank_lx, ank_ly, ank_rx, ank_ry = marker_coords
    eom = eom.col_join(sm.Matrix(marker_eqs))

#breakpoint()
print('Number of operations in eom:', sm.count_ops(eom))

states = symbolics.states
num_states = len(states)

# %%
# make an initial guess from the standing solution
logging.info('Making an initial guess.')
fname = 'standing.csv'
if not os.path.exists(fname):
    # executes standing solution and generates 'standing.csv'
    import solve_standing
standing_sol = np.loadtxt(fname)
standing_state = standing_sol[0:num_states - 1].reshape(-1, 1)  # coordinates and speeds as column vector
state_traj = np.tile(standing_state, (1, num_nodes))  # make num_nodes copies
delta_traj = h*np.arange(num_nodes).reshape(1, -1)    # row vector
tor_traj = np.zeros((num_angles, num_nodes))          # intialize torques to zero
mar_traj = np.zeros((len(marker_coords), num_nodes))          # intialize torques to zero
initial_guess = np.concatenate((state_traj, delta_traj, tor_traj, mar_traj))  # complete trajectory
initial_guess = initial_guess.flatten()               # make a single row vector
np.random.seed(1)  # this makes the result reproducible
initial_guess = initial_guess + 0.01*np.random.random_sample(initial_guess.size)

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
bounds.update({k: (-1000.0, 1000.0)
               for k in [Tb, Tc, Td, Te, Tf, Tg]})

# %%
# To enforce a half period, set the right leg's angles at the initial time to
# be equal to the left leg's angles at the final time and vice versa. The same
# goes for the joint angular rates.
#
instance_constraints = (
    qax.func(0*h) - 0.0,
    qax.func(duration) - 0.0,
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
    # torques must also be periodic, because torques at t=0 are never
    # used with Backward Euler, and will become zero due to the cost function
    Tb.func(0*h) - Te.func(duration),
    Tc.func(0*h) - Tf.func(duration),
    Td.func(0*h) - Tg.func(duration),
    Te.func(0*h) - Tb.func(duration),
    Tf.func(0*h) - Tc.func(duration),
    Tg.func(0*h) - Td.func(duration),
)

if TRACK_MARKERS:
    marker_instance_constraints = (
        ank_ly.func(0*h) - ank_ry.func(duration),
        ank_lx.func(0*h) - ank_rx.func(duration),
        ank_ry.func(0*h) - ank_ly.func(duration),
        ank_rx.func(0*h) - ank_lx.func(duration),
    )
    instance_constraints += marker_instance_constraints

# %%
# The objective is a combination of squared torques and squared tracking errors

# Make indices for the free variables that are angles and torques.
# The final node is excluded, it is the first node of the next cycle
angle_indices = np.empty(num_angles*(num_nodes-1), dtype=np.int64)
inodes = np.arange(0, num_nodes - 1)
for iangle in range(0, num_angles):
    # skip the first 3 DOFs and angles before iangle
    angle_indices[iangle*(num_nodes - 1) + inodes] = ((3 + iangle)*num_nodes +
                                                      inodes)

torque_indices = np.empty(num_angles*(num_nodes - 1), dtype=np.int64)
inodes = np.arange(0, num_nodes - 1)
for itorque in range(0, num_angles):
    # skip the state trajectories, and torques before itorque
    torque_indices[itorque*(num_nodes-1) + inodes] = (
        (num_states+itorque)*num_nodes + inodes)


def obj(prob, free, obj_show=False):
    f_torque = (1e-6*obj_Wtorque*np.sum(free[torque_indices]**2)/
                torque_indices.size)
    f_track = (obj_Wtrack*np.sum((free[angle_indices] - ang_data)**2)/
               angle_indices.size)
    f_total = f_torque + f_track

    if TRACK_MARKERS:
        for var, lab in zip((ank_lx, ank_ly, ank_rx, ank_ry),
                            ('LLM.PosX', 'LLM.PosY', 'RLM.PosX', 'RLM.PosY')):
            vals = prob.extract_values(var, free)
            # we only return 49 nodes from measured, so add first to last
            meas_vals = np.hstack((marker_df[lab].values,
                                   marker_df[lab].values[0]))
            if 'X' in lab:
                meas_vals = meas_vals - 0.12
            else:
                meas_vals = meas_vals - 0.04
            # TODO : Ton divides the whole angle track by num_angles*num_nodes,
            # need to combine this division for angle and marker track.
            f_total += (obj_Wtrack*np.sum((vals - meas_vals)**2)/len(vals)/
                        len(marker_coords))

    if obj_show:
        print(f"   obj: {f_total:.3f} = {f_torque:.3f}(torque) + "
              f"{f_track:.3f}(track)")

    return f_total


def obj_grad(prob, free):
    grad = np.zeros_like(free)
    grad[torque_indices] = (2e-6*obj_Wtorque*free[torque_indices]/
                            torque_indices.size)
    grad[angle_indices] = (2.0*obj_Wtrack*(free[angle_indices] - ang_data)/
                           angle_indices.size)

    if TRACK_MARKERS:
        for var, lab in zip((ank_lx, ank_ly, ank_rx, ank_ry),
                            ('LLM.PosX', 'LLM.PosY', 'RLM.PosX', 'RLM.PosY')):
            vals = prob.extract_values(var, free)
            meas_vals = np.hstack((marker_df[lab].values,
                                   marker_df[lab].values[0]))
            if 'X' in lab:
                meas_vals = meas_vals - 0.12
            else:
                meas_vals = meas_vals - 0.04
            prob.fill_free(grad, var, 2.0*obj_Wtrack*(vals - meas_vals)/
                           len(vals)/len(marker_coords))

    return grad


# %%
# create the optimization problem
logging.info('Creating the opty problem.')

# Create a belt velocity signal v(t)
traj_map = {
    v: walking_speed*np.ones(num_nodes),
}

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
)
prob.add_option('max_iter', 3000)
prob.add_option('tol', 1e-3)
prob.add_option('constr_viol_tol', 1e-4)
prob.add_option('print_level', 0)


# %%
# solve the gait optimization problem for given belt speed
def solve_gait(speed, initial_guess=None):

    # change the belt speed signal
    traj_map[v] = speed*np.ones(num_nodes)

    # solve
    print(datetime.now().strftime("%H:%M:%S") +
          f" solving for {speed:.3f} m/s")
    solution, info = prob.solve(initial_guess)
    # we accept solve_succeeded (0) and solved_to_acceptable_level (1)
    if info['status'] < 0:
        print("IPOPT was not successful.")
        #breakpoint()

    # show the final objective function value and its contributions
    obj(prob, solution, obj_show=True)

    return solution


# solve for a series of increasing speeds, ending at the required speed
for speed in np.linspace(0.0, walking_speed, num=10):
    solution = solve_gait(speed, initial_guess)
    initial_guess = solution  # use this solution as guess for the next problem

# %%
# make plots

# extract angles and torques
ang = solution[angle_indices].reshape(num_angles, num_nodes-1).transpose()
tor = solution[torque_indices].reshape(num_angles, num_nodes-1).transpose()
dat = ang_data.reshape(num_angles, num_nodes-1).transpose()

# construct a right side full gait cycle trajectory
ang = np.rad2deg(np.vstack((ang[:, 0:3], ang[:, 3:6], ang[1, 0:3])))
tor = np.vstack((tor[:, 0:3], tor[:, 3:6], tor[1, 0:3]))
dat = np.rad2deg(np.vstack((dat[:, 0:3], dat[:, 3:6], dat[1, 0:3])))
t = np.arange(2*num_nodes-1) * h

# use Winter's sign convention (knee flexion angle
# and hip/ankle extension torque)
ang[:, 1] = -ang[:, 1]
dat[:, 1] = -dat[:, 1]
tor[:, [0, 2]] = -tor[:, [0, 2]]

plot_joint_comparison(t, ang, tor, dat)


def plot_ankle():
    fig, ax = plt.subplots()
    ax.plot(prob.extract_values(ank_lx, solution),
            prob.extract_values(ank_ly, solution), color='C0',
            label='Model, Left')
    ax.plot(marker_df['LLM.PosX'] - 0.12, marker_df['LLM.PosY'] - 0.04,
            color='C0', linestyle='--', label='Data, Left')
    ax.plot(prob.extract_values(ank_rx, solution),
            prob.extract_values(ank_ry, solution), color='C1',
            label='Model, Right')
    ax.plot(marker_df['RLM.PosX'] - 0.12, marker_df['RLM.PosY'] - 0.04,
            color='C1', linestyle='--', label='Data, Right')
    ax.set_aspect("equal")
    ax.legend()
    return ax


if TRACK_MARKERS:
    plot_ankle()


plt.show()


def animate():

    ground, origin = symbolics.inertial_frame, symbolics.origin
    trunk, rthigh, rshank, rfoot, lthigh, lshank, lfoot = symbolics.segments

    fig = plt.figure(figsize=(10.0, 4.0))

    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)

    # hip_proj = origin.locatenew('m', qax*ground.x)
    # scene = Scene3D(ground, hip_proj, ax=ax3d)
    scene = Scene3D(ground, origin, ax=ax3d)

    # creates the stick person
    scene.add_line([
        rshank.joint,
        rfoot.toe,
        rfoot.heel,
        rshank.joint,
        rthigh.joint,
        trunk.joint,
        trunk.mass_center,
        trunk.joint,
        lthigh.joint,
        lshank.joint,
        lfoot.heel,
        lfoot.toe,
        lshank.joint,
    ], color="k")

    # creates a moving ground (many points to deal with matplotlib limitation)
    # ?? can we make the dashed line move to the left?
    scene.add_line([origin.locatenew('gl', s*ground.x) for s in
                    np.linspace(-2.0, 2.0)], linestyle='--', color='tab:green',
                   axlim_clip=True)

    # adds CoM and unit vectors for each body segment
    for seg in symbolics.segments:
        scene.add_body(seg.rigid_body)

    # show ground reaction force vectors at the heels and toes, scaled to
    # visually reasonable length
    scene.add_vector(contact_force(rfoot.toe, ground, origin, v)/600.0,
                     rfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(rfoot.heel, ground, origin, v)/600.0,
                     rfoot.heel, color="tab:blue")
    scene.add_vector(contact_force(lfoot.toe, ground, origin, v)/600.0,
                     lfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(lfoot.heel, ground, origin, v)/600.0,
                     lfoot.heel, color="tab:blue")

    scene.lambdify_system(states + symbolics.specifieds + symbolics.constants)
    gait_cycle = np.vstack((
        xs,  # q, u shape(2n, N)
        speed + np.zeros((1, len(times))),  # belt speed shape(1, N)
        rs[:6, :],  # r, shape(q, N)
        np.repeat(np.atleast_2d(np.array(list(par_map.values()))).T,
                  len(times), axis=1),  # p, shape(r, N)
    ))
    scene.evaluate_system(*gait_cycle[:, 0])

    scene.axes.set_proj_type("ortho")
    scene.axes.view_init(90, -90, 0)
    scene.plot(prettify=False)

    ax3d.set_xlim((-0.8, 0.8))
    ax3d.set_ylim((-0.2, 1.4))
    ax3d.set_aspect('equal')
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.set_ticklabels([])
        axis.set_ticks_position("none")

    eval_rforce = sm.lambdify(
        states + symbolics.specifieds + symbolics.constants,
        (contact_force(rfoot.toe, ground, origin, v) +
            contact_force(rfoot.heel, ground, origin, v)).to_matrix(ground),
        cse=True)

    eval_lforce = sm.lambdify(
        states + symbolics.specifieds + symbolics.constants,
        (contact_force(lfoot.toe, ground, origin, v) +
            contact_force(lfoot.heel, ground, origin, v)).to_matrix(ground),
        cse=True)

    rforces = np.array([eval_rforce(*gci).squeeze() for gci in gait_cycle.T])
    lforces = np.array([eval_lforce(*gci).squeeze() for gci in gait_cycle.T])

    ax2d.plot(times, rforces[:, :2], times, lforces[:, :2])
    ax2d.grid()
    ax2d.set_ylabel('Force [N]')
    ax2d.set_xlabel('Time [s]')
    ax2d.legend(['Horizontal GRF (r)', 'Vertical GRF (r)',
                 'Horizontal GRF (l)', 'Vertical GRF (l)'], loc='upper right')
    ax2d.set_title('Foot Ground Reaction Force Components')
    vline = ax2d.axvline(times[0], color='black')

    def update(i):
        scene.evaluate_system(*gait_cycle[:, i])
        scene.update()
        vline.set_xdata([times[i], times[i]])
        return scene.artists + (vline,)

    ani = FuncAnimation(
        fig,
        update,
        frames=range(len(times)),
        interval=h*1000,
    )

    return ani


# %%
# Use symmeplot to make an animation of the motion.
if make_animation:
    xs, rs, _ = prob.parse_free(solution)
    times = np.linspace(0.0, (num_nodes - 1)*h, num=num_nodes)

    animation = animate()

    animation.save('human_gait.gif', fps=int(1.0/h))

    plt.show()
