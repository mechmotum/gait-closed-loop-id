"""
Generates a half cycle of normal gait, by minimizing a combination of tracking
and error.

The resulting state trajectory will be used as a reference trajectory for a
known feedback controller to generate synthetic data for testing our controller
identification method.
"""
from ctypes.util import find_library
import platform
import pickle

from opty import Problem
from pygait2d.segment import time_symbol
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    ANG_GAIT2D_COLS,
    GRF_GAIT2D_COLS,
    TOR_GAIT2D_COLS,
    animate,
    create_bounds,
    create_instance_constraints,
    create_known_parameters_map,
    extract_flattened_meas_arrays,
    extract_solution,
    extract_values,
    extract_values_diff,
    fill_free,
    full_gait_from_half,
    generate_symbolic_eqs,
    load_measurement_data,
    logger,
    plot_joint_comparison,
    plot_marker_comparison,
    print_settings,
    solve_gait,
    gen_initial_guess_from_standing,
)

# Primary settings (capitalize)
settings = {
    "EOM_SCALE": 10.0,  # scaling factor for eom
    "GAIT_CYCLE_NUM": 0,  # gait cycle to select from measurment data
    "GENFORCE_SCALE": 0.001,  # convert to kN and kNm
    # passed to IPOPT: mumps, spral, ma57, ma77, ma86, ma97
    "LINEAR_SOLVER": 'mumps' if find_library('coinhsl') is None else 'ma57',
    "MAKE_ANIMATION": True,
    "NUM_NODES": 51,  # number of time nodes for the half period
    "SAVE_RESULT": False,
    "SEED": True,  # set to integer value for specific seed value, True(=1), or False
    "STIFFNESS_EXP": 2,  # exponent of the contact stiffness force
    "SUBJECT_MASS": 70.0,  # kg of subject from trial 20, TODO: extract from metadata
    "USE_WINTER_DATA": False,  # if we want to track Winter's gait data
    # For weights, remove parts of the objective by setting to integer 0.
    "WANG": 1000.0,  # weight of mean squared angle tracking error (in rad)
    "WGRF": 0.0005,  # weight of mean squared GRF tracking error (in Newtons)
    "WMAR": 0,  # weight of mean squared marker tracking error (in meters)
    "WREG": 1e-6,  # weight of mean squared time derivatives
    "WTOR": 1000.0,  # weight of the mean squared torque (in kNm) objective
}


def obj(prob, free, obj_show=False):
    """Returns the objective scalar value given the values of the free
    optimization variables.

    Objective function::

        J = WTOR*mean(joint_torque**2)
          + WANG*mean(joint_angle_error**2)
          + WREG*mean((dx/dt)**2)
          + WMAR*mean(marker_error**2)
          + WGRF*mean(grf_error**2)

    The final node is excluded from all means. Due to symmetry and periodicity
    constraints, it is the mirror image of the first node, and we don't want to
    include it twice.

    Parameters
    ==========
    prob : opty.Problem
        Instantiated optimal control problem.
    free : ndarray, shape(num_free,)
        Values of the free optimization variables.
    obj_show : boolean, optional
        If true, the objective components will be printed.

    Returns
    =======
    f_tot : float
        Scalar objective value.

    """
    # NOTE : slice(0, -1) is used to avoid double counting the periodic
    # duplicate values

    # minimize mean joint torque
    tor_vals = extract_values(prob, free, *syms.joint_torques, slice=(0, -1))
    f_tor = 1e-6*settings['WTOR']*np.sum(tor_vals**2)/len(tor_vals)

    f_tot = f_tor

    # minimize mean angle tracking error
    if settings['WANG'] != 0:
        ang_vals = extract_values(prob, free, *syms.joint_angles,
                                  slice=(0, -1))
        f_ang = settings['WANG']*np.sum((ang_vals - ang_meas)**2)/len(ang_vals)
        f_tot += f_ang

    # smooth all regularization trajectories
    if settings['WREG'] != 0:
        diff = extract_values_diff(prob, free, *reg_syms)
        f_reg = settings['WREG']*np.sum(diff**2)/h**2/len(diff)
        f_tot += f_reg

    # minimize mean marker tracking error
    if settings['WMAR'] != 0:
        # vals -> shape(num_markers*(num_nodes - 1), 1)
        mar_vals = extract_values(prob, free, *msyms.marker_syms, slice=(0, -1))
        f_mar = settings['WMAR']*np.sum((mar_vals - mar_meas)**2)/len(mar_vals)
        f_tot += f_mar

    # minimize mean ground reaction force tracking error
    if settings['WGRF'] != 0:
        grf_vals = extract_values(prob, free, *msyms.grf_syms, slice=(0, -1))
        f_grf = settings['WGRF']*np.sum((grf_vals - grf_meas)**2)/len(grf_vals)
        f_tot += f_grf

    if obj_show:
        msg = (f"   obj: {f_tot:.3f} = {f_tor:.3f}(torque)")
        if settings['WREG'] != 0:
            msg += f" + {f_reg:.3f}(reg)"
        if settings['WANG'] != 0:
            msg += f" + {f_ang:.3f}(angle)"
        if settings['WMAR'] != 0:
            msg += f" + {f_mar:.3f}(marker)"
        if settings['WGRF'] != 0:
            msg += f" + {f_grf:.3f}(grf)"
        print(msg)

    return f_tot


def obj_grad(prob, free):
    """Returns the gradient of the objective function with respect to the free
    optimization variables.

    Parameters
    ==========
    prob : opty.Problem
        Instantiated optimal control problem.
    free : ndarray, shape(num_free,)
        Values of the free optimization variables.

    Returns
    =======
    grad : ndarray, shape(free,)
        Gradient of the objective.

    """
    grad = np.zeros_like(free)

    tor_vals = extract_values(prob, free, *syms.joint_torques, slice=(0, -1))
    fill_free(prob, grad, 2e-6*settings['WTOR']*tor_vals/len(tor_vals),
              *syms.joint_torques, slice=(0, -1))

    if settings['WANG'] != 0:
        ang_vals = extract_values(prob, free, *syms.joint_angles,
                                  slice=(0, -1))
        fill_free(prob, grad,
                  2.0*settings['WANG']*(ang_vals - ang_meas)/len(ang_vals),
                  *syms.joint_angles, slice=(0, -1))
    if settings['WREG'] != 0:
        # NOTE : The regularization should be added on top of the tor_vals and
        # ang_vals.
        diff = extract_values_diff(prob, free, *reg_syms)
        reg_grad = settings['WREG']*2.0*diff/h**2/len(diff)
        # NOTE : Add twice with a shift for correct gradient.
        fill_free(prob, grad, -reg_grad, *reg_syms, slice=(None, -1), add=True)
        fill_free(prob, grad, reg_grad, *reg_syms, slice=(1, None), add=True)

    if settings['WMAR'] != 0:
        mar_vals = extract_values(prob, free, *msyms.marker_syms,
                                  slice=(0, -1))
        fill_free(prob, grad,
                  2.0*settings['WMAR']*(mar_vals - mar_meas)/len(mar_vals),
                  *msyms.marker_syms, slice=(0, -1))

    if settings['WGRF'] != 0:
        grf_vals = extract_values(prob, free, *msyms.grf_syms, slice=(0, -1))
        fill_free(prob, grad,
                  2.0*settings['WGRF']*(grf_vals - grf_meas)/len(grf_vals),
                  *msyms.grf_syms, slice=(0, -1))

    return grad


if __name__ == "__main__":

    print_settings(settings)

    df, duration, h, walking_speed = load_measurement_data(settings)
    syms, msyms = generate_symbolic_eqs(settings)
    reg_syms = syms.states + syms.joint_torques
    ang_meas, mar_meas, grf_meas = extract_flattened_meas_arrays(settings, df,
                                                              msyms)
    par_map = create_known_parameters_map(settings, syms)
    v = syms.specifieds[-1]
    traj_map = {
        v: df['Speed'].values,  # shape(settings['NUM_NODES'],)
    }
    instance_constraints = create_instance_constraints(settings, syms, msyms,
                                                       duration, h)
    bounds = create_bounds(syms)

    logger.info('Creating the opty problem.')
    prob = Problem(
        obj,
        obj_grad,
        msyms.eom,
        syms.states,
        settings['NUM_NODES'],
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
    # unique solution. We therefore require qax(t=0) = 0 (it would be better to
    # use the bounds to require free[0]=zero, but I don't know how to do this
    # in opty, I only see bounds on the whole trajectory)
    if settings['WMAR'] == 0:
        # instance_constraints += (qax.func(0*h),)
        prob.lower_bound[0] = 0.0
        prob.upper_bound[0] = 0.0

    if settings['LINEAR_SOLVER'].startswith('ma'):
        # TODO : Load correct shared lib on Windows/Darwin
        if platform.system() == 'Linux':
            prob.add_option('hsllib', 'libcoinhsl.so')
        prob.add_option('linear_solver', settings['LINEAR_SOLVER'])
        if settings['LINEAR_SOLVER'] == 'ma57':
            prob.add_option('ma57_pivot_order', 2)

    prob.add_option('max_iter', 3000)
    prob.add_option('tol', 1e-3)
    prob.add_option('constr_viol_tol', 1e-4)
    prob.add_option('print_level', 0)
    #prob.add_option('derivative_test', 'first-order')

    initial_guess = gen_initial_guess_from_standing(settings, syms, msyms)

    # solve for a series of increasing speeds, ending at the required speed
    for speed_scale in np.linspace(0.1, 1.0, num=10):
        speeds = speed_scale*df['Speed'].values
        solution = solve_gait(prob, v, speeds, initial_guess)
        initial_guess = solution  # use as guess for the next problem

    # extract angles and torques (simulation and measured)
    ang_sol, tor_sol, grf_sol = extract_solution(syms, prob, solution)
    ang_meas = df[ANG_GAIT2D_COLS].values[:-1, :]  # shape(N-1, 6)
    tor_meas = df[TOR_GAIT2D_COLS].values[:-1, :]  # shape(N-1, 6)
    grf_meas = df[GRF_GAIT2D_COLS].values[:-1, :]  # shape(N-1, 4)

    # construct a right side full gait cycle trajectory
    plot_time = np.linspace(0.0, duration - h, num=2*settings['NUM_NODES'] - 1)
    ang_sol_full, tor_sol_full, grf_sol_full = full_gait_from_half(
        ang_sol, tor_sol, grf_sol)
    ang_meas_full, tor_meas_full, grf_meas_full = full_gait_from_half(
        ang_meas, tor_meas, grf_meas)

    # Generate plots and animations
    plot_joint_comparison(plot_time, ang_sol_full, tor_sol_full, ang_meas_full,
                        torques_meas=tor_meas_full, grf=grf_sol_full,
                        grf_meas=grf_meas_full)

    if settings['WMAR'] != 0:
        plot_marker_comparison(msyms.marker_syms, msyms.marker_labels, df,
                               prob, solution)

    plt.show()

    if settings['SAVE_RESULT']:
        result = settings.copy()
        result.update({
            'ang_sol_full': ang_sol_full,
            'tor_sol_full': tor_sol_full,
            'grf_sol_full': grf_sol_full,
            'solution': solution,
            'h': h,
            'duration': duration,
            'walking_speed': walking_speed,
        })
        with open('result.pkl', 'wb') as f:
            # TODO : Generate a unique filename based on the settings.
            pickle.dump(result, f)

    if settings['MAKE_ANIMATION']:
        animation = animate(syms, prob, solution, traj_map[v], grf_meas)
        animation.save('human_gait.gif', fps=int(1.0/h))
        plt.show()
