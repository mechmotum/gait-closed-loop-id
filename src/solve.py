# solve.py

# Generates a half cycle of normal gait, by minimizing a combination
# of tracking and error.

# The resulting state trajectory will be used as a reference trajectory
# for a known feedback controller to generate synthetic data for
# testing our controller identification method.

# Import all necessary modules, functions, and classes:
import os
from datetime import datetime
from opty import Problem
from opty.utils import f_minus_ma
from pygait2d import simulate
from pygait2d.segment import time_symbol
from symmeplot.matplotlib import Scene3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sympy as sm
from derive import derive_equations_of_motion, contact_force

# %%
# some settings
make_animation = True
num_nodes = 50       # number of time nodes for the half period
eom_scale = 1.0      # scaling factor for eom
obj_Wtorque = 0.001  # weight of torque objective
obj_Wtrack = 10.0    # weight of tracking objective

# load normal gait data from Winter's book
fname = os.path.join(os.path.dirname(__file__),
                     os.path.join('..', 'data', 'Winter_normal.csv'))
data = np.genfromtxt(fname, delimiter=',')

# extract gait cycle duration and speed
duration = data[1, 2]/2  # half gait cycle duration
h = duration/(num_nodes - 1)
walking_speed = data[2, 2]

# extract  hip, knee, ankle angle (full gait cycle)
ang = np.deg2rad(data[6:57, 4:7])
# invert Winter's knee angle, to be compatible with our model
ang[:, 1] = -ang[:, 1]

# convert full gait cycle (one side) into a half gait cycle for both sides and
# resample to num_nodes
ang = np.concatenate((ang[:26, :], ang[25:, :]), axis=1)
rows, num_angles = ang.shape
ang_resampled = np.zeros((num_nodes, num_angles))
t = np.arange(0, rows)/(rows-1)  # gait phase from data
t_new = np.arange(0, num_nodes)/(num_nodes-1)  # gait phase for simulation
for i in range(0, num_angles):
    ang_resampled[:, i] = np.interp(t_new, t, ang[:, i])

# store the angle trajectories in a 1d array, for tracking
ang_data = ang_resampled.transpose().flatten()

# %%
# Derive the equations of motion
print(datetime.now().strftime("%H:%M:%S"))
symbolics = derive_equations_of_motion()

mass_matrix = symbolics[0]
forcing_vector = symbolics[1]
constants = symbolics[3]
coordinates = symbolics[4]
speeds = symbolics[5]
specified = symbolics[6]

eom = eom_scale*f_minus_ma(mass_matrix, forcing_vector, coordinates + speeds)
print('Number of operations in eom:', sm.count_ops(eom))

# %%
# Create a state variable "delt" which is equal to time, for use in controller
# and/or instance constraints.
#
# .. math::
#
#    \Delta_t(t) = \int_{t_0}^{t} d\tau
#
delt = sm.Function('delt', real=True)(time_symbol)
eom = eom.col_join(sm.Matrix([delt.diff(time_symbol) - 1]))

states = coordinates + speeds + [delt]
num_states = len(states)

# %%
# make an initial guess from the standing solution
print(datetime.now().strftime("%H:%M:%S")+' making initial guess.')
fname = 'standing.csv'
if not os.path.exists(fname):
    # executes standing solution and generates 'standing.csv'
    import solve_standing
standing_sol = np.loadtxt(fname)
standing_state = standing_sol[0:2*(num_states - 1):2].reshape(-1, 1)  # coordinates and speeds as column vector
state_traj = np.tile(standing_state, (1, num_nodes))  # make num_nodes copies
delta_traj = h*np.arange(num_nodes).reshape(1, -1)    # row vector
tor_traj = np.zeros((num_angles, num_nodes))          # intialize torques to zero
initial_guess = np.concatenate((state_traj, delta_traj, tor_traj))  # complete trajectory
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
qax, qay, qa, qb, qc, qd, qe, qf, qg = coordinates
uax, uay, ua, ub, uc, ud, ue, uf, ug = speeds
v, Tb, Tc, Td, Te, Tf, Tg            = specified

# %%
# The constants are loaded from a file of realistic geometry, mass, inertia,
# and foot deformation properties of an adult human.
par_map = simulate.load_constants(constants,
                                  os.path.join(os.path.dirname(__file__), '..',
                                               'data/example_constants.yml'))

# %%
# Bound all the states to human realizable ranges.
#
# - The trunk should stay generally upright and be at a possible walking
#   height.
# - Only let the hip, knee, and ankle flex and extend to realistic limits.
# - Put a maximum on the peak torque values.
bounds = {
    delt: (0.0, 2.0),
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
    delt.func(0*h) - 0.0,
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


# %%
# The objective is a combination of squared torques and squared tracking errors

# Make indices for the free variables that are angles and torques
# TODO: this could be made less specific for this model and problem
# TODO: we should exclude the torques and angles at the first (or last) time point,
# otherwise those are counted twice. We can't do that for the torques (yet)
# because, with backward Euler, torque at t=0 is never used, so requires an
# instance constraint (which does not work right now, see instance constraints).
angle_indices =  3*num_nodes + np.arange(0, num_angles*num_nodes)  # start at the hip angle
torque_indices = num_states*num_nodes + np.arange(0, num_angles*num_nodes)


def obj(free, obj_show=False):
    f_torque = obj_Wtorque*h*np.sum(free[torque_indices]**2)
    f_track = obj_Wtrack*h*np.sum((free[angle_indices] - ang_data)**2)
    f_total = f_torque + f_track
    if obj_show:
        print(f"   obj: {f_total:.3f} = {f_torque:.3f}(torque) + {f_track:.3f}(track)")
    return f_total


def obj_grad(free):
    grad = np.zeros_like(free)
    grad[torque_indices] = 2.0*obj_Wtorque*h*free[torque_indices]
    grad[angle_indices] = 2.0*obj_Wtrack *h*(free[angle_indices] - ang_data)
    return grad


# %%
# create the optimization problem
print(datetime.now().strftime("%H:%M:%S") + " creating the opty problem")

# Create a belt velocity signal v(t)
traj_map = {v: np.zeros(num_nodes)}

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
prob.add_option('print_level', 0)


# %%
# solve the gait optimization problem for given belt speed
def solve_gait(speed, initial_guess=None):

    # change the belt speed signal
    traj_map[v] = speed + np.zeros(num_nodes)

    # solve
    print(datetime.now().strftime("%H:%M:%S") +
          f" solving for {speed:.3f} m/s")
    solution, info = prob.solve(initial_guess)
    # we accept solve_succeeded (0) and solved_to_acceptable_level (1)
    if info['status'] > 1:
        print("IPOPT was not successful.")
        breakpoint()

    # show the final objective function value and its contributions
    obj(solution, obj_show=True)

    return solution


# solve for a series of increasing speeds, ending at the required speed
# initial_guess = None
for speed in np.linspace(0, walking_speed, 10):
    solution = solve_gait(speed, initial_guess)
    #initial_guess = solution # use this solution as initial guess for the next problem
    initial_guess = solution + np.random.normal(scale=0.1, size=len(solution)) # use this solution as initial guess for the next problem

fname = f'human_gait_{num_nodes}_nodes_solution.csv'
np.savetxt(fname, solution, fmt='%.5f')

# %%
# make plots

# extract angles and torques
ang = solution[angle_indices].reshape(num_angles, num_nodes).transpose()
tor = solution[torque_indices].reshape(num_angles, num_nodes).transpose()
dat = ang_data.reshape(num_angles, num_nodes).transpose()

# construct a right side full gait cycle trajectory
ang = np.rad2deg(np.concatenate((ang[:, 0:3],ang[1:, 3:6]), axis=0))
tor =            np.concatenate((tor[:, 0:3],tor[1:, 3:6]), axis=0)
dat = np.rad2deg(np.concatenate((dat[:, 0:3],dat[1:, 3:6]), axis=0))
t = np.arange(2*num_nodes-1) * h

# use Winter's sign convention (knee flexion angle
# and hip/ankle extension torque)
ang[:, 1] = -ang[:, 1]
dat[:, 1] = -dat[:, 1]
tor[:, [0, 2]] = -tor[:, [0, 2]]
anglabels = ('hip flexion', 'knee flexion', 'ankle dorsiflexion')
torlabels = ('hip extension', 'knee extension', 'ankle plantarflexion')

# plot
plt.figure(figsize=(6.0, 9.0))
colors = ('r', 'b', 'g')

plt.subplot(2, 1, 1)
for i in range(3):
    plt.plot(t, ang[:, i], colors[i], label=anglabels[i])
    plt.plot(t, dat[:, i], colors[i]+'--')
plt.legend()
plt.ylabel('angle (deg)')

plt.subplot(2, 1, 2)
for i in range(3):
    plt.plot(t, tor[:, i], colors[i], label=torlabels[i])
plt.legend()
plt.ylabel('torque (Nm)')
plt.xlabel('time (s)')

plt.show()

# %%
# Use symmeplot to make an animation of the motion.
if make_animation:
    xs, rs, _ = prob.parse_free(solution)
    times = np.linspace(0.0, (num_nodes - 1)*h, num=num_nodes)


    def animate():

        ground, origin, segments = symbolics[8], symbolics[9], symbolics[10]
        trunk, rthigh, rshank, rfoot, lthigh, lshank, lfoot = segments

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
        for seg in segments:
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

        scene.lambdify_system(states + specified + constants)
        gait_cycle = np.vstack((
            xs,  # q, u shape(2n, N)
            speed + np.zeros((1, len(times))),  # belt speed shape(1, N)
            rs,  # r, shape(q, N)
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
            states + specified + constants,
            (contact_force(rfoot.toe, ground, origin, v) +
             contact_force(rfoot.heel, ground, origin, v)).to_matrix(ground),
            cse=True)

        eval_lforce = sm.lambdify(
            states + specified + constants,
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


    animation = animate()

    animation.save('human_gait.gif', fps=int(1.0/h))

    plt.show()
