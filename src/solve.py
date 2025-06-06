# solve.py

# Generates a half cycle of normal gait, by minimizing a combination
# of tracking and error.

# The resulting state trajectory will be used as a reference trajectory
# for a known feedback controller to generate synthetic data for
# testing our controller identification method.

# Import all necessary modules, functions, and classes:
import os
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
treadmill = False
num_nodes = 50         # number of time nodes for the half period
eom_scale = 0.1       # this seems to solve a lot faster than scale=1
obj_scale = 0.001       # this seems to solve a lot faster than scale=1

# load normal gait data from Winter's book
fname = os.path.join(os.path.dirname(__file__), '../data/Winter_normal.csv')
data = np.genfromtxt(fname, delimiter=',')

# extract gait cycle duration and speed
duration = data[1,2]/2;  # extract half gait cycle duration
h = duration / (num_nodes - 1)
speed = data[2,2];  # walking speed
if treadmill:
    treadmill_speed = speed
    required_translation = 0
else:
    treadmill_speed = 0;
    required_translation = duration*speed

# extract  hip,knee,ankle angle (full gait cycle)
ang      = np.deg2rad(data[6:57,4:7])
ang[:,1] = -ang[:,1]  # invert Winter's knee angle, to be compatible with our model

# convert full gait cycle (one side) into a half gait cycle for both sides
# and resample to num_nodes
ang = np.concatenate((ang[:26,:],ang[25:,:]), axis=1)
rows,cols = ang.shape;
ang_resampled = np.zeros((num_nodes,cols))
t = np.arange(0,rows) / (rows-1)  # gait phase from data
t_new = np.arange(0,num_nodes) / (num_nodes-1) # gait phase for simulation
for i in range(0,cols):
    ang_resampled[:,i] = np.interp(t_new, t, ang[:,i])

# store the angle trajectories in a 1d array, for tracking, and generate
# corresponding indices for the free variables in the optimization problem
# ang_data = reshape()

# %%
# Derive the equations of motion
symbolics = derive_equations_of_motion()

mass_matrix = symbolics[0]
forcing_vector = symbolics[1]
constants = symbolics[3]
coordinates = symbolics[4]
speeds = symbolics[5]
specified = symbolics[6]

eom = eom_scale * f_minus_ma(mass_matrix, forcing_vector, coordinates + speeds)

# %%
# The equations of motion have this many mathematical operations:
print('Number of operations in eom:', sm.count_ops(eom))

# %%
# Create a state variable "delt" which is equal to time, for use in
# controller and/or instance constraints.
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
# Set belt velocity v(t) to a constant speed
traj_map = {
    v: treadmill_speed + np.zeros(num_nodes),
}

# %%
# Bound all the states to human realizable ranges.
#
# - The trunk should stay generally upright and be at a possible walking
#   height.
# - Only let the hip, knee, and ankle flex and extend to realistic limits.
# - Put a maximum on the peak torque values.

bounds = {
    delt: (0.0, 10.0),
    qax: (0.0, 10.0),
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
bounds.update({k: (-200.0, 200.0)
               for k in [Tb, Tc, Td, Te, Tf, Tg]})

# %%
# The average speed can be fixed by constraining the total distance traveled.
# To enforce a half period, set the right leg's angles at the initial time to
# be equal to the left leg's angles at the final time and vice versa. The same
# goes for the joint angular rates.
#
instance_constraints = (
    delt.func(0*h) - 0.0,
    qax.func(0*h) - 0.0,
    qax.func(duration) - required_translation,
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
)


# %%
# The objective is to minimize the mean of all joint torques.
def obj(free):
    """Minimize the sum of the squares of the control torques."""
    T = free[num_states*num_nodes:]
    return obj_scale * h*np.sum(T**2)


def obj_grad(free):
    grad = np.zeros_like(free)
    T = free[num_states*num_nodes:]
    grad[num_states*num_nodes:] = 2.0*h*T
    return obj_scale * grad


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
    parallel=True,
)
prob.add_option('max_iter', 3000)

# %%
# Find the optimal solution and save it if it converges.
#
# This loads a precomputed solution (if it exists) to use as initial guess.
# Delete the file to use a random initial guess.
fname = f'human_gait_{num_nodes}_nodes_solution.csv'
if os.path.exists(fname):
    print('Loading solution stored in {} as the initial guess.'.format(fname),
          'Delete the file now to use a random guess')
    input("Hit Enter to continue, or CTRL-C to quit.")
    initial_guess = np.loadtxt(fname)
else:
    np.random.seed(1)  # this makes the result reproducible
    # a random intial guess that stays close to the midpoint between bounds
    initial_guess = (0.5*(prob.lower_bound + prob.upper_bound) +
                     0.01*(prob.upper_bound - prob.lower_bound)*
                     np.random.random_sample(prob.num_free))
solution, info = prob.solve(initial_guess)
if info['status'] == 0:
    np.savetxt(fname, solution, fmt='%.5f')
else:
    breakpoint()
    
# %%
# make a plot of joint angles


# %%
# Use symmeplot to make an animation of the motion.
xs, rs, _ = prob.parse_free(solution)
times = np.linspace(0.0, (num_nodes - 1)*h, num=num_nodes)
h_val = h

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
        treadmill_speed + np.zeros((1, len(times))),  # belt speed shape(1, N)
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
        interval=h_val*1000,
    )

    return ani


animation = animate()

animation.save('human_gait.gif', fps=int(1.0/h_val))

plt.show()
