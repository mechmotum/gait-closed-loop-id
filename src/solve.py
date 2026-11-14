"""
Generates a half cycle of normal gait, by minimizing a combination of tracking
and error.

The resulting state trajectory will be used as a reference trajectory for a
known feedback controller to generate synthetic data for testing our controller
identification method.
"""
from ctypes.util import find_library

from pygait2d.segment import time_symbol
import matplotlib.pyplot as plt
import numpy as np

from utils import (
    animate,
    build_full_gait_cycles,
    create_bounds,
    create_instance_constraints,
    create_known_parameters_map,
    extract_flattened_meas_arrays,
    gen_initial_guess_from_standing,
    generate_symbolic_eqs,
    load_measurement_data,
    obj,
    obj_grad,
    plot_joint_comparison,
    plot_marker_comparison,
    print_settings,
    save_result,
    setup_opty_problem,
    solve_gait,
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


if __name__ == "__main__":

    print_settings(settings)

    # 1. Generate all input data for the opty problem.
    df, duration, h, walking_speed = load_measurement_data(settings)
    syms, msyms = generate_symbolic_eqs(settings)
    ang_meas_f, mar_meas_f, grf_meas_f = extract_flattened_meas_arrays(
        settings, df, msyms)
    par_map = create_known_parameters_map(settings, syms)
    v = syms.specifieds[-1]
    traj_map = {
        v: df['Speed'].values,  # shape(settings['NUM_NODES'],)
    }
    instance_constraints = create_instance_constraints(settings, syms, msyms,
                                                       duration, h)
    bounds = create_bounds(syms)

    # 2. Generate the opty problem.
    prob = setup_opty_problem(settings, obj, obj_grad, syms, msyms, h, par_map,
                              traj_map, instance_constraints, bounds,
                              time_symbol, ang_meas_f, mar_meas_f, grf_meas_f)

    # 3. Solve the problem.
    initial_guess = gen_initial_guess_from_standing(settings, syms, msyms)

    # solve for a series of increasing speeds, ending at the required speed
    for speed_scale in np.linspace(0.1, 1.0, num=10):
        speeds = speed_scale*df['Speed'].values
        solution = solve_gait(prob, v, speeds, initial_guess)
        initial_guess = solution  # use as guess for the next problem

    # 4. Display and record the results.
    # TODO : This function has too many inputs and outputs.
    (ang_meas, tor_meas, grf_meas, plot_time, ang_sol_full, tor_sol_full,
     grf_sol_full, ang_meas_full, tor_meas_full, grf_meas_full) = \
        build_full_gait_cycles(settings, syms, prob, solution, df, duration, h)

    plot_joint_comparison(plot_time, ang_sol_full, tor_sol_full, ang_meas_full,
                        torques_meas=tor_meas_full, grf=grf_sol_full,
                        grf_meas=grf_meas_full)

    if settings['WMAR'] != 0:
        plot_marker_comparison(msyms.marker_syms, msyms.marker_labels, df,
                               prob, solution)

    plt.show()

    if settings['SAVE_RESULT']:
        save_result(settings, ang_sol_full, tor_sol_full, grf_sol_full,
                    solution, h, duration, walking_speed)

    if settings['MAKE_ANIMATION']:
        animation = animate(syms, prob, solution, traj_map[v], grf_meas)
        animation.save('human_gait.gif', fps=int(1.0/h))
        plt.show()
