"""

Load data from multiple subsequent gait cycles.

Load, for example, the Winter data and interpolate it for each of the gait
cycles in the data with respect to the percent gait cycle. This should create a
series of gait cycles stretched in time to match each gait cycle in the
measurement data. This will be used as the initial guess for the optimizer.

The instance constraints should not have periodicty constraints. So, compared
to the half gait cycle solutions, there will be no instance constraints.

Do we need to do the iterative solutions from 0 speed to walking speed?

"""

from ctypes.util import find_library

import numpy as np
from pygait2d.segment import time_symbol

from utils import (
    ANG_GAIT2D_COLS,
    TOR_GAIT2D_COLS,
    create_bounds,
    create_known_parameters_map,
    generate_symbolic_eqs,
    load_sample_data,
    load_winter_data_frame,
    match_winter_to_meas,
    obj,
    obj_grad,
    print_settings,
    setup_opty_problem,
)

conf = {
    "EOM_SCALE": 10.0,  # scaling factor for eom
    "GAIT_CYCLE_NUM": 0,  # gait cycle to select from measurment data
    "GENFORCE_SCALE": 0.001,  # convert to kN and kNm
    "IPOPT_CONSTR_VIOL_TOL": 1e-4,
    "IPOPT_LINEAR_SOLVER": ('mumps' if find_library('coinhsl') is None else
                            'ma57'),
    "IPOPT_MAX_ITER": 3000,
    "IPOPT_PRINT_LEVEL": 5,
    "IPOPT_TOL": 1e-3,
    "MAKE_ANIMATION": True,
    "NUM_GAIT_CYCLES": 2,
    "NUM_NODES": 201,
    "SAVE_RESULT": False,
    "SEED": True,  # set to integer value for specific seed value
    "STIFFNESS_EXP": 2,  # exponent of the contact stiffness force
    "SUBJECT_MASS": 70.0,  # kg of subject from trial 20
    "USE_WINTER_DATA": False,  # if we want to track Winter's gait data
    # For weights, remove parts of the objective by setting to integer 0.
    "WANG": 1000.0,  # weight of mean squared angle tracking error (in rad)
    "WGRF": 0,  # weight of mean squared GRF tracking error (in Newtons)
    "WMAR": 0,  # weight of mean squared marker tracking error (in meters)
    "WREG": 1e-6,  # weight of mean squared time derivatives
    "WTOR": 1000.0,  # weight of the mean squared torque (in kNm) objective
}

print_settings(conf)

# multiple gait cycles
meas_df = load_sample_data(
    conf['NUM_NODES'],
    first_cycle=0,
    last_cycle=conf['NUM_GAIT_CYCLES'] - 1,
    half_cycle=False
)

# single gait cycle
wint_df = load_winter_data_frame(
    conf['NUM_NODES']//conf['NUM_GAIT_CYCLES'],
    half_cycle=False,
)

# Winter's data repeated and interpolated to align with the measurement data.
# TODO : Figure out why there are odd (non monotonic seeming) points in
# init_df.
init_df = match_winter_to_meas(meas_df, wint_df)

syms, msyms = generate_symbolic_eqs(conf)

par_map = create_known_parameters_map(conf, syms)
traj_map = {syms.specifieds[-1]: meas_df['Speed'].values}
instance_constraints = None
bounds = create_bounds(syms)

duration = meas_df['Time'].values[-1]
h = duration/(conf['NUM_NODES'] - 1)

ang_meas_f = meas_df[ANG_GAIT2D_COLS].values[:-1, :].transpose().flatten()
if conf['WMAR'] != 0:
    mar_meas_f = meas_df[msyms.marker_labels].values[:-1, :].T.flatten()
else:
    mar_meas_f = None
if conf['WGRF'] != 0:
    grf_meas_f = meas_df[msyms.grf_labels].values[:-1, :].T.flatten()
else:
    grf_meas_f = None

# (qax(t), qay(t), qa(t), qb(t), qc(t), qd(t), qe(t), qf(t), qg(t), uax(t),
# uay(t), ua(t), ub(t), uc(t), ud(t), ue(t), uf(t), ug(t))
first_init = np.zeros((3, conf['NUM_NODES']))
ang_init = init_df[ANG_GAIT2D_COLS].values.transpose()
rat_init = np.zeros((9, conf['NUM_NODES']))
tor_init = init_df[TOR_GAIT2D_COLS].values.transpose()
initial_guess = np.concatenate((first_init, ang_init, rat_init, tor_init))
if conf['WMAR'] != 0:
    mar_init = np.zeros((len(msyms.marker_syms), len(meas_df)))
    initial_guess = np.concatenate((initial_guess, mar_init))
if conf['WGRF'] != 0:
    grf_init = init_df[msyms.grf_labels].values.transpose()
    initial_guess = np.concatenate((initial_guess, grf_init))
initial_guess = initial_guess.flatten()
if conf['SEED']:
    np.random.seed(conf['SEED'])
initial_guess = (initial_guess +
                 0.01*np.random.random_sample(len(initial_guess)))

# TODO : obj & obj_grad avoid double counting period solution, that needs to be
# removed for this solve.
prob = setup_opty_problem(conf, obj, obj_grad, syms, msyms, h, par_map,
                          traj_map, instance_constraints, bounds, time_symbol,
                          ang_meas_f, mar_meas_f, grf_meas_f)
solution, info = prob.solve(initial_guess)
