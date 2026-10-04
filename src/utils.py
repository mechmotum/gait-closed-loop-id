import os

import numpy as np
import pandas as pd
import sympy as sm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pygait2d.segment import time_varying
from symmeplot.matplotlib import Scene3D
from matplotlib.animation import FuncAnimation

GAITFILE = '020-longitudinal-perturbation-gait-cycles.csv'
CALIBFILE = '020-calibration-pose.csv'
DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GAITDATAPATH = os.path.join(DATADIR, GAITFILE)
CALIBDATAPATH = os.path.join(DATADIR, CALIBFILE)


def tile_standing(standing_sol, num_nodes, num_angles, num_states):
    """Returns an array with the standing solution repeated for every node.

    Parameters
    ==========
    standing_sol : ndarray, shape(num_free, )
    num_nodes : integer
    num_angles : integer
    num_states : integer

    Returns
    =======
    trajectory : ndarray, shape(num_free, num_nodes)

    """
    # coordinates and speeds as column vector
    standing_state = standing_sol[0:num_states].reshape(-1, 1)
    # make num_nodes copies
    state_traj = np.tile(standing_state, (1, num_nodes))
    # intialize torques to zero
    tor_traj = np.zeros((num_angles, num_nodes))
    # complete trajectory
    return np.concatenate((state_traj, tor_traj))


def fill_free(problem, free, values, *variables, slice=(None, None),
              add=False):
    """Replaces the values in a vector shaped the same as the free optimization
    vector corresponding to the variable names.

    Parameters
    ==========
    problem : Problem
    free : ndarray, shape(n*N + q*N + r + s, )
        Vector to replace values in.
    values : ndarray, shape(N,) or float
        Numerical values to insert, arrays for each variable must be in
        order of monotonic time and then stacked in order variables. The
        shape depends on how many variables and whether they are
        trajectories or parameters.
    varables: Symbol or Function()(time)
        One or more of the unknown optimization variables in the problem.
    slice : tuple of integers
        If provided this will allow you to select the same subset of bookended
        slices of time from all variables. If you want state x but only want to
        return the first half of the simulation you can do ``slice=(None,
        num_nodes//2)`` which translates to ``x[None:num_nodes//2]``.
    add : boolean, optional
        If true the values will be added to the existing values in free instead
        of being overwritten.

    """
    d = problem._extraction_indices
    idxs = []
    for var in variables:
        try:
            idxs += d[var][slice[0]:slice[1]]
        except KeyError:
            raise ValueError(f'{var} not an unknown in this problem.')
    if add:
        free[idxs] += values
    else:
        free[idxs] = values


def extract_values_diff(problem, free, *variables):
    N = problem.collocator.num_collocation_nodes
    num_vars = len(variables)
    # shape(num_vars*N, )
    vals = problem.extract_values(free, *variables)
    # shape(num_vars, N - 1)
    vals_diff = np.diff(vals.reshape(num_vars, N))
    # shape(num_vars*(N - 1), )
    return vals_diff.flatten()


def extract_values(problem, free, *variables, slice=(None, None)):
    """Returns the numerical values of the free variables.

    Parameters
    ==========
    problem : Problem
    free : ndarray, shape(n*N + q*N + r + s)
        The free optimization vector of the system, required if var is an
        unknown optimization variable.
    variables : Symbol or Function()(time), len(d)
        One or more of the known or unknown variables in the problem.
    slice : tuple of integers
        If provided this will allow you to select the same subset of bookended
        slices of time from all variables. If you want state x but only want to
        return the first half of the simulation you can do ``slice=(None,
        num_nodes//2)`` which translates to ``x[None:num_nodes//2]``.

    Returns
    =======
    values : ndarray
        The numerical values of the variables. The shape depends on how
        many variables and whether they are trajectories or parameters.

    """
    d = problem._extraction_indices
    idxs = []
    for var in variables:
        try:
            idxs += d[var][slice[0]:slice[1]]
        except KeyError:
            raise ValueError(f'{var} not an unknown in this problem.')
    return free[idxs]


class SymbolDict(dict):
    """A mapping from SymPy symbols or functions of time to arbitrary values.
    Values can alternatively be retrieved using the string name of the symbol
    or function of time."""

    def __getitem__(self, key):
        if isinstance(key, str):
            keys = [sym for sym in self.keys() if sym.name == key]
            if len(keys) != 1:
                raise KeyError('Not found or symbols with same names.')
            key = keys[0]
        val = dict.__getitem__(self, key)
        return val

    def __setitem__(self, key, val):
        if isinstance(key, str):
            keys = [sym for sym in self.keys() if sym.name == key]
            if len(keys) != 1:
                raise KeyError('Not found or symbols with same names.')
            key = keys[0]
        dict.__setitem__(self, key, val)


def animate(symbolics, xs, rs, h, speed, times, par_map, stiffness_exp):

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
    grf = symbolics.ground_reaction_forces
    scene.add_vector(grf['Right Foot toe']/600.0, rfoot.toe,
                     color="tab:blue")
    scene.add_vector(grf['Right Foot heel']/600.0, rfoot.heel,
                     color="tab:blue")
    scene.add_vector(grf['Left Foot toe']/600.0, lfoot.toe,
                     color="tab:blue")
    scene.add_vector(grf['Left Foot heel']/600.0, lfoot.heel,
                     color="tab:blue")

    scene.lambdify_system(symbolics.states + symbolics.specifieds +
                          symbolics.constants)
    gait_cycle = np.vstack((
        xs,  # q, u shape(2n, N)
        rs[:6, :],  # r, shape(q, N)
        speed + np.zeros((1, len(times))),  # belt speed shape(1, N)
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
        symbolics.states + symbolics.specifieds + symbolics.constants,
        (grf['Right Foot heel'] + grf['Right Foot toe']).to_matrix(ground),
        cse=True)

    eval_lforce = sm.lambdify(
        symbolics.states + symbolics.specifieds + symbolics.constants,
        (grf['Left Foot heel'] + grf['Left Foot toe']).to_matrix(ground),
        cse=True)

    rforces = np.array([eval_rforce(*gci).squeeze() for gci in gait_cycle.T])
    lforces = np.array([eval_lforce(*gci).squeeze() for gci in gait_cycle.T])

    full_cycle_times = np.hstack((times, times[-1] + times))
    full_gait_cycle = np.hstack((gait_cycle, gait_cycle))
    ax2d.plot(full_cycle_times,
              np.vstack((rforces[:, :2], lforces[:, :2])), color='C0')
    ax2d.plot(full_cycle_times,
              np.vstack((lforces[:, :2], rforces[:, :2])), color='C1')
    ax2d.grid()
    ax2d.set_ylabel('Force [N]')
    ax2d.set_xlabel('Time [s]')
    ax2d.legend(['Horizontal GRF (r)', 'Vertical GRF (r)',
                 'Horizontal GRF (l)', 'Vertical GRF (l)'], loc='upper right')
    ax2d.set_title('Foot Ground Reaction Force Components')
    vline = ax2d.axvline(times[0], color='black')

    def update(i):
        scene.evaluate_system(*full_gait_cycle[:, i])
        scene.update()
        vline.set_xdata([full_cycle_times[i], full_cycle_times[i]])
        return scene.artists + (vline,)

    ani = FuncAnimation(
        fig,
        update,
        frames=range(len(full_cycle_times)),
        interval=h*1000,  # milliseconds
    )

    return ani


def get_sym_by_name(iterable, *sym_strs):
    """Returns SymPy Symbols or Function()(t)s that matches the provided
    string.

    Examples
    ========

    >>> import sympy as sm
    >>> t = sm.symbols('t')
    >>> a, b, c = sm.symbols('a, b, c', cls=sm.Function)
    >>> get_sym_by_name([a(t), b(t), c(t)], 'c')
    c(t)
    >>> get_sym_by_name([a(t), b(t), c(t)], 'c', 'a')
    (c(t), a(t))

    """
    syms = []
    for sym_str in sym_strs:
        for s in iterable:
            if sym_str == s.name:
                syms.append(s)
                break
    if len(syms) == len(sym_strs):
        if len(syms) == 1:
            return syms[0]
        else:
            return tuple(syms)
    else:
        raise ValueError('One or more sym_strs not in iterable.')


def generate_marker_equations(symbolics):
    """Returns the equations for the x and y coordinates of markers to track.

    Parameters
    ==========
    symbolics : pygait2d.derive.Symbolics
        Dataclass containing the symbolic model.

    Returns
    =======
    variables : list of Symbol
        SymPy symbols for the x and y coordinate of each marker.
    equations : list of Expr
        SymPy expressions representing the equations for the x and y
        coordiantes of each marker.
    data_labels : list of str
        List of measured marker labels that correspond to the model points to
        track.

    """

    O, N = symbolics.origin, symbolics.inertial_frame
    trunk, rthigh, rshank, rfoot, lthigh, lshank, lfoot = symbolics.segments
    fyd, fyg = get_sym_by_name(symbolics.constants, 'fyd', 'fyg')

    # NOTE : Code will only work if these are uncommented in L/R pairs.
    # The heel and toe markers are above the ground by the distance 0.56*fyd,
    # 0.56*fyg. The 0.56 = HEE.PosY/LM.PosY in the calibration pose and is
    # estimate from the subject in trial 20 with these numbers:
    # RHEE.PosY.mean() = 0.051276430563978154
    # RLM.PosY.mean() = 0.09135660837737156
    # TODO : Make the heel/toe marker height a variable and store it in the
    # subject specific calibration.
    p = 0.56
    points = {
        'ank_l': lshank.joint,  # left ankle
        'ank_r': rshank.joint,  # right ankle
        'hel_l': lfoot.heel.locatenew('hee_l', -p*fyg*lfoot.reference_frame.y),
        'hel_r': rfoot.heel.locatenew('hee_r', -p*fyd*rfoot.reference_frame.y),
        'hip_l': trunk.joint,  # hip
        'hip_r': trunk.joint,  # hip
        'kne_l': lthigh.joint,  # left knee
        'kne_r': rthigh.joint,  # right knee
        'toe_l': lfoot.toe.locatenew('toe_l', -p*fyg*lfoot.reference_frame.y),
        'toe_r': rfoot.toe.locatenew('toe_r', -p*fyd*rfoot.reference_frame.y),
    }

    point_data_map = {
        'ank_l': 'LLM',
        'ank_r': 'RLM',
        'hel_l': 'LHEE',
        'hel_r': 'RHEE',
        'hip_l': 'LGTRO',
        'hip_r': 'RGTRO',
        'kne_l': 'LLEK',
        'kne_r': 'RLEK',
        'toe_l': 'LMT5',
        'toe_r': 'RMT5',
    }

    variables = []
    equations = []
    data_labels = []

    for lab, point in points.items():
        x, y = time_varying(f'{lab}x, {lab}y')
        variables += [x, y]
        x_eq = x - point.pos_from(O).dot(N.x)
        y_eq = y - point.pos_from(O).dot(N.y)
        equations += [x_eq, y_eq]
        data_labels += [point_data_map[lab] + '.PosX',
                        point_data_map[lab] + '.PosY']

    return variables, equations, data_labels

def generate_grf_equations(symbolics):
    """Returns the equations for the x and y ground reaction forces to track.

    Parameters
    ==========
    symbolics : pygait2d.derive.Symbolics
        Dataclass containing the symbolic model.

    Returns
    =======
    variables : list of Symbol
        SymPy symbols for the x and y measure numbers of the resultant force on
        each foot.
    equations : list of Expr
        SymPy expressions representing the equations for the x and y measure
        numbers of the resultant force on each foot.
    labels : list of str
        List of measured ground reaction force labels that correspond to the
        model ground reaction forces to track.

    """
    grf = symbolics.ground_reaction_forces
    N = symbolics.inertial_frame

    right_vars = sm.Matrix(time_varying('Frx, Fry'))
    left_vars = sm.Matrix(time_varying('Flx, Fly'))

    variables = right_vars.col_join(left_vars)[:]

    right_force = grf['Right Foot heel'] + grf['Right Foot toe']
    left_force = grf['Left Foot heel'] + grf['Left Foot toe']

    right_eqs = right_vars - right_force.to_matrix(N)[:2, :]
    left_eqs = left_vars - left_force.to_matrix(N)[:2, :]
    equations = right_eqs.col_join(left_eqs)

    # FP1 is left
    # FP2 is right
    labels = [
        'FP2.ForX',
        'FP2.ForY',
        'FP1.ForX',
        'FP1.ForY',
    ]

    return variables, equations, labels


def extract_gait_cycle(df, number):
    """Returns a single gait cycle as a data frame from a measurement data
    frame based on the gait cycle number."""

    if number not in df['major'].values:
        msg = '{} not in {}-{}'
        raise ValueError(msg.format(number, df['major'].min(),
                                    df['major'].max()))

    return df[df['major'] == number]


def plot_points(df):
    """Returns a plot axis showing a 2D view of the primary markers defining
    the walker moving through the gait cycle."""

    marker_labels = [
        'LSHO',
        'LGTRO',
        'LLEK',
        'LLM',
        'LHEE',
        'LTOE',
        'RSHO',
        'RGTRO',
        'RLEK',
        'RLM',
        'RHEE',
        'RTOE',
    ]

    fig, ax = plt.subplots()

    for i, lab in enumerate(marker_labels):
        if lab not in ['LSHO', 'RSHO']:
            x1 = df[lab + '.PosX'].values[0:-1:4]
            y1 = df[lab + '.PosY'].values[0:-1:4]
            x2 = df[marker_labels[i - 1] + '.PosX'].values[0:-1:4]
            y2 = df[marker_labels[i - 1] + '.PosY'].values[0:-1:4]

            ax.plot(np.vstack((x1, x2)), np.vstack((y1, y2)), color='black',
                    alpha=0.5)
        if lab.startswith('R'):
            color = 'C0'
        else:
            color = 'C1'
        ax.plot(df[lab + '.PosX'], df[lab + '.PosY'], color=color)

    ax.set_aspect('equal')

    return ax


def load_winter_data_frame(num_nodes=None, half_cycle=False,
                           drop_last_node=False):
    """Returns Winter's normative gait data transformed to match naming
    conventions of the Gait2D model.

    Parameters
    ==========
    num_nodes : integer, optional
        If provided, data will be interpolated at the number of linearly spaced
        time nodes - 1.
    half_cycle : boolean, optional
        If true, returns the first half of the gait cycle 0% to 50%. Else the
        full gait cycle 0% to 100% is returned.
    drop_last_node : boolean, optional
        If true, then the returned data frame excludes the 50% or 100% node.

    Returns
    =======
    df : DataFrame, shape(num_nodes or num_nodes - 1, 19)
        Index is range(num_nodes) for `drop_last_node=False` or range(num_nodes
        - 1) for `drop_last_node=True`. Column names are:

        1. 'Percent Gait Cycle'
        2. 'Time'
        3. 'Speed'
        4. 'FP2.ForY' (right, vertical)
        5. 'FP1.ForY' (left, vertical)
        6. 'FP2.ForX' (right, longitudinal)
        7. 'FP1.ForX' (left, longitudinal)
        8. 'Right.Hip.Flexion.Angle'
        9. 'Left.Hip.Flexion.Angle'
        10. 'Right.Knee.Extension.Angle'
        11. 'Left.Knee.Extension.Angle'
        12. 'Right.Ankle.DorsiFlexion.Angle'
        13. 'Left.Ankle.DorsiFlexion.Angle'
        14. 'Right.Hip.Flexion.Moment'
        15. 'Left.Hip.Flexion.Moment'
        16. 'Right.Knee.Extension.Moment'
        17. 'Left.Knee.Extension.Moment'
        18. 'Right.Ankle.DorsiFlexion.Moment'
        19. 'Left.Ankle.DorsiFlexion.Moment'

    """
    winter_moore_map = {
        # FP2 is the right force plate
        'vertical GRF': ('FP2.ForY', 1.0, 0.0),
        'horizontal GRF': ('FP2.ForX', 1.0, 0.0),
        'hip angle': ('Right.Hip.Flexion.Angle', 1.0, 0.0),
        # Winters' knee is flexion, negate to extension
        'knee angle': ('Right.Knee.Extension.Angle', -1.0, 0.0),
        'ankle angle': ('Right.Ankle.DorsiFlexion.Angle', 1.0, 0.0),
        # Winters' hip is extension, negate to flexion
        'hip moment': ('Right.Hip.Flexion.Moment', -1.0, 0.0),
        'knee moment': ('Right.Knee.Extension.Moment', 1.0, 0.0),
        # Winters' ankle is plantarflexion, negate to dorsiflexion
        'ankle moment': ('Right.Ankle.DorsiFlexion.Moment', -1.0, 0.0),
    }

    subject_mass = 75.0  # kg (from Winter's book)

    # notes on Winter_normal.csv:
    # rows 0, 1, 2, 3 at not tabular
    # last row is empty
    # angles are in degrees
    # forces and torques are normalized by the mass of the person
    fname = os.path.join(DATADIR, 'Winter_normal.csv')

    # extract gait cycle duration and speed
    data = np.genfromtxt(fname, delimiter=',')
    duration = data[1, 2]
    walking_speed = data[2, 2]

    df = pd.read_csv(fname, header=4, skiprows=[5], index_col='sample')
    # last row is empty
    df.drop(index=df.index[-1], inplace=True)
    # two empty columns
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3'], inplace=True)
    df.index = df.index.astype(int)
    # remove whitespace
    df.columns = df.columns.str.strip()
    # Time is inclusive [0%, 100%] for full gait cycle duration.
    df['Time'] = np.linspace(0.0, duration, num=len(df))
    df['Speed'] = walking_speed
    kinetics = ['horizontal GRF', 'vertical GRF', 'hip moment', 'knee moment',
                'ankle moment']
    for kinetic in kinetics:
        df[kinetic] = df[kinetic]*subject_mass

    for k, v in winter_moore_map.items():
        name, sign, offset = v
        df[name] = sign*df[k] + offset
        # The Winter data has 51 data points from 0% to 100% gait cycle for
        # right leg starting at heel contact.
        #
        # sample 0 = 0%, right heel contact
        # sample 25 = 50%
        # sample 50 = 100%, right heel contact
        #
        # i  | right | i       | left
        # 0  | 0%    | 25      | 50%
        # 1  | 2%    | 26      | 52%
        # .  | .     | .       | .
        # 24 | 48%   | 49      | 98%
        # 25 | 50%   | 0 or 50 | 0% or 100%
        # 26 | 52%   | 1       | 2%
        # 27 | 54%   | 2       | 4%
        # .  | .     | .       | .
        # 49 | 98%   | 24      | 48%
        # 50 | 100%  | 25      | 50%
        #
        # When creating the full gait cycle for the left side you have a choice
        # of putting the right's i=0 or i=50 at the left's heel strike. Or you
        # could average the values.
        # Two choices:
        #left = np.hstack((df[k][25:], df[k][1:26]))  # drop 0
        left = np.hstack((df[k][25:-1], df[k][:26]))  # drop 50
        lname = name.replace('Right', 'Left').replace('FP2', 'FP1')
        df[lname] = sign*left + offset

    df.rename(columns={'%gait cycle': 'Percent Gait Cycle'}, inplace=True)

    for k in winter_moore_map.keys():
        del df[k]

    if half_cycle:
        # returns [0% to 50%] inclusive
        df = df.iloc[:26, :]

    if num_nodes is not None:
        t0, tf = df['Time'].values[[0, -1]]
        new_time = np.linspace(t0, tf, num=num_nodes)
        df = pd.DataFrame(interp1d(df['Time'], df.values, axis=0)(new_time),
                          columns=df.columns,
                          index=np.arange(len(new_time)))

    if drop_last_node:
        return df.iloc[:-1, :]
    else:
        return df


def load_winter_data(num_nodes, as_data_frame=False):
    """Returns interpolated normative gait data from Winter's book formulated
    as a gait cycle of both legs from 0% to ``50%*(1 - 1/(N - 1))`` and
    matching Gait2D's angle and moment sign convention.

    Parameters
    ==========
    num_nodes : int
        Desired number of time nodes N for 50% of the gait cycle.
    as_data_frame : boolean
        If true, ``ang_data`` is a Pandas data frame that includes time,
        percent gait cycle, and named columns.

    Returns
    =======
    duration : float
        Time in seconds corresponding to the duration of 50% of the gait cycle.
    walking_speed : float
        Average walking speed in meters per second.
    num_angles : int
        Number of angles: 6. (r & l hip, knee, ankle)
    ang_data : ndarray, shape((num_nodes-1)*num_angles,) or DataFrame
        Angle data in radians linear interpolated at the times corresponding to
        the number of nodes in the sign convention of our simulation model::

            [rhip0, ..., rhipN-2,  # flexion
             rknee0, ..., rkneeN-2,  # extension
             rankle0, ..., rankleN-2,  # dorsiflexion
             lhip0, ..., lhipN-2,  # flexion
             lknee0, ..., lkneeN-2,  # extension
             lankle0, ..., lankleN-2]  # dorsiflexion

    """
    # NOTE : Gait2D model has angular hip flexion, knee extension, and ankle
    # dorsi flexion as positive.
    fname = os.path.join(DATADIR, 'Winter_normal.csv')

    data = np.genfromtxt(fname, delimiter=',')

    # extract gait cycle duration and speed
    duration = data[1, 2]/2  # half gait cycle duration, 0%-50% inclusive
    walking_speed = data[2, 2]

    # extract hip, knee, ankle angle (full gait cycle)
    # NOTE : Winters' ankle angle = 0 reprsents nominal standing config.
    ang = np.deg2rad(data[6:57, 4:7])
    kin = data[6:57, 7:12]  # [horizontal, vertical, hip, knee, ankle]
    # invert Winter's knee angle, to be compatible with our model
    ang[:, 1] = -ang[:, 1]
    # invert Winters' hip and ankle moments to make them flexion & dorsiflexion
    kin[:, 2] = -kin[:, 2]
    kin[:, 4] = -kin[:, 4]
    # convert to N from N/kg
    kin = 75.0*kin  # 75.0 kg from Winters' book

    # convert full gait cycle (one side) into a half gait cycle for both sides
    # and resample to num_nodes; take first 26 for the right and last 26 for
    # the left (note that 50% node is present in both slices)
    ang = np.concatenate((ang[:26, :], ang[25:, :]), axis=1)  # shape(26, 6)
    kin = np.concatenate((kin[:26, :], kin[25:, :]), axis=1)  # shape(26, 10)
    rows, num_angles = ang.shape
    t = np.arange(0, rows)/(rows - 1)  # [0, ..., 1], shape(26,)
    # t_new: [0, ..., 1 - 1/(N-1)], shape(25,)
    t_new = np.arange(0, num_nodes - 1)/(num_nodes - 1)
    # ang_resampled shape(time, [hip, knee, ankle, hip, knee, ankle])
    ang_resampled = interp1d(t, ang, axis=0)(t_new)
    kin_resampled = interp1d(t, kin, axis=0)(t_new)

    if as_data_frame:
        ang_data = pd.DataFrame(ang_resampled, columns=[
            'Right.Hip.Flexion.Angle',
            'Right.Knee.Extension.Angle',
            'Right.Ankle.DorsiFlexion.Angle',
            'Left.Hip.Flexion.Angle',
            'Left.Knee.Extension.Angle',
            'Left.Ankle.DorsiFlexion.Angle'
        ])
        percent_step = 50.0/(num_nodes - 1)
        percent = np.linspace(0.0, 50.0 - percent_step, num=num_nodes - 1)
        ang_data['Percent Gait Cycle'] = percent
        ang_data['Time'] = t_new
        kin_data = pd.DataFrame(kin_resampled, columns=[
            'FP2.ForX',
            'FP2.ForY',
            'Right.Hip.Flexion.Moment',
            'Right.Knee.Extension.Moment',
            'Right.Ankle.DorsiFlexion.Moment',
            'FP1.ForX',
            'FP1.ForY',
            'Left.Hip.Flexion.Moment',
            'Left.Knee.Extension.Moment',
            'Left.Ankle.DorsiFlexion.Moment',
        ])
        ang_data = pd.concat((ang_data, kin_data), axis=1)
    else:
        # store the angle trajectories in a 1d array, for tracking
        ang_data = ang_resampled.transpose().flatten()

    return duration, walking_speed, num_angles, ang_data


def load_sample_data(num_nodes, gait_cycle_number=10):
    """Returns interpolated data from a measurement file formulated as a single
    gait cycle of both legs from 0% to ``50%*(1 - 1/(N - 1))`` in the same
    format as ``load_winter_data()`` (Gait2D sign conventions).

    Parameters
    ==========
    num_nodes : int
        Desired number of time nodes N for [0%, 50%] of the gait cycle.
    gait_cycle_number: integer, optional
        Number from 0 to ``total gait cycles - 1``. Use this to select one of
        many gait cycles in the data file.

    Returns
    =======
    duration : float
        Time in seconds corresponding to the duration of 50% of the gait cycle.
    walking_speed : float
        Average walking speed in meters per second.
    num_angles : int
        Numer of angles: 6. (r & l hip, knee, ankle)
    ang_data : ndarray, shape((num_nodes-1)*num_angles,)
        Angle data in radians linear interpolated at the times corresponding to
        the number of nodes::

            [rhip0, ..., rhipN-2,  # flexion
             rknee0, ..., rkneeN-2,  # extension
             rankle0, ..., rankleN-2,  # dorsiflexion
             lhip0, ..., lhipN-2,  # flexion
             lknee0, ..., lkneeN-2,  # extension
             lankle0, ..., lankleN-2]  # dorsiflexion

    mark_df : DataFrame
        Data frame containing the marker trajectories.
    kinetic_df : DataFrame
        Contains the kinetic (forces, moments) trajectories. Postive moments
        for gait2d model are:

            - hip flexion
            - knee extension
            - ankle dorsiflexion

    ang_df : DataFrame
        Contains the joint angle trajectores in the Gait2D sign convention.
    time : ndarray, shape(num_nodes - 1,)
        Time values corresponding to the output gait data.
    percent : ndarray, shape(num_nodes - 1,)
        Gait cycle percent values corresponding to the output gait data.

    """
    df = extract_gait_cycle(pd.read_csv(GAITDATAPATH), gait_cycle_number)
    full_time = df['Original Time'].values
    full_time = full_time - full_time[0]
    full_percent = df['Percent Gait Cycle'].values  # [0%, ..., 100%)
    # interpolate to find the half cycle duration (at exactly 50%)
    duration = np.interp(0.5, full_percent, full_time)

    # TODO : Extract from metadata so other trials can be used.
    walking_speed = 1.2  # nominal speed from trial 20 meta data

    angles = [
        'Right.Hip.Flexion.Angle',
        'Right.Knee.Flexion.Angle',
        'Right.Ankle.PlantarFlexion.Angle',
        'Left.Hip.Flexion.Angle',
        'Left.Knee.Flexion.Angle',
        'Left.Ankle.PlantarFlexion.Angle',
    ]
    all_angles = angles + [
        'Right.Knee.Extension.Angle',
        'Right.Ankle.DorsiFlexion.Angle',
        'Left.Knee.Extension.Angle',
        'Left.Ankle.DorsiFlexion.Angle',
    ]
    ang_arr = -df[angles].values.copy()  # change to extension (knee and ankle)
    ang_arr[:, [0, 3]] *= -1  # change hip back to flexion
    ang_arr[:, [2, 5]] -= np.pi/2  # shift ankle 90 degrees

    # Adds angles needed for simulation comparison.
    df['Right.Knee.Extension.Angle'] = -df['Right.Knee.Flexion.Angle']
    df['Right.Ankle.DorsiFlexion.Angle'] = -df['Right.Ankle.PlantarFlexion.Angle'] - np.pi/2
    df['Right.Ankle.PlantarFlexion.Angle'] = df['Right.Ankle.PlantarFlexion.Angle'] + np.pi/2
    df['Left.Knee.Extension.Angle'] = -df['Left.Knee.Flexion.Angle']
    df['Left.Ankle.DorsiFlexion.Angle'] = -df['Left.Ankle.PlantarFlexion.Angle'] - np.pi/2
    df['Left.Ankle.PlantarFlexion.Angle'] = df['Left.Ankle.PlantarFlexion.Angle'] + np.pi/2
    ang_vals = df[all_angles].values.copy()

    markers = [
        'LGTRO.PosX',
        'LGTRO.PosY',
        'LHEE.PosX',
        'LHEE.PosY',
        'LLEK.PosX',
        'LLEK.PosY',
        'LLM.PosX',
        'LLM.PosY',
        'LMT5.PosX',
        'LMT5.PosY',
        'LSHO.PosX',
        'LSHO.PosY',
        'LTOE.PosX',
        'LTOE.PosY',
        'RGTRO.PosX',
        'RGTRO.PosY',
        'RHEE.PosX',
        'RHEE.PosY',
        'RLEK.PosX',
        'RLEK.PosY',
        'RLM.PosX',
        'RLM.PosY',
        'RMT5.PosX',
        'RMT5.PosY',
        'RSHO.PosX',
        'RSHO.PosY',
        'RTOE.PosX',
        'RTOE.PosY',
    ]
    marker_vals = df[markers].values.copy()

    kinetics = [
        'FP1.ForX',
        'FP1.ForY',
        'FP1.ForZ',
        'FP2.ForX',
        'FP2.ForY',
        'FP2.ForZ',
        'Right.Hip.Flexion.Moment',
        'Right.Knee.Flexion.Moment',
        'Right.Ankle.PlantarFlexion.Moment',
        'Left.Hip.Flexion.Moment',
        'Left.Knee.Flexion.Moment',
        'Left.Ankle.PlantarFlexion.Moment',
    ]
    all_kinetics = kinetics + [
        'Right.Knee.Extension.Moment',
        'Right.Ankle.DorsiFlexion.Moment',
        'Left.Knee.Extension.Moment',
        'Left.Ankle.DorsiFlexion.Moment',
    ]
    df['Right.Knee.Extension.Moment'] = -df['Right.Knee.Flexion.Moment']
    df['Right.Ankle.DorsiFlexion.Moment'] = -df['Right.Ankle.PlantarFlexion.Moment']
    df['Left.Knee.Extension.Moment'] = -df['Left.Knee.Flexion.Moment']
    df['Left.Ankle.DorsiFlexion.Moment'] = -df['Left.Ankle.PlantarFlexion.Moment']
    kinetic_vals = df[all_kinetics].values.copy()

    # NOTE : It is fraught to use arange() for constructing these due to
    # numerical stability of arange(), use linspace()!
    time_step = duration/(num_nodes - 1)
    time = np.linspace(0.0, duration - time_step, num=num_nodes - 1)
    percent_step = 50.0/(num_nodes - 1)
    percent = np.linspace(0.0, 50.0 - percent_step, num=num_nodes - 1)

    interp_ang_arr = interp1d(full_time, ang_arr, axis=0)(time)
    interp_ang_vals = interp1d(full_time, ang_vals, axis=0)(time)
    interp_mark_arr = interp1d(full_time, marker_vals, axis=0)(time)
    interp_kinetic_arr = interp1d(full_time, kinetic_vals, axis=0)(time)

    mark_df = pd.DataFrame(dict(zip(markers, interp_mark_arr.T)))
    kinetic_df = pd.DataFrame(dict(zip(all_kinetics, interp_kinetic_arr.T)))
    ang_df = pd.DataFrame(dict(zip(all_angles, interp_ang_vals.T)))

    return (duration, walking_speed, len(angles), interp_ang_arr.T.flatten(),
            mark_df, kinetic_df, ang_df, time, percent)


def plot_joint_comparison(t, angles, torques, angles_meas, torques_meas=None,
                          grf=None, grf_meas=None):
    """Plots the optimization solution alongside the measurement data.

    Parameters
    ==========
    t : array_like, shape(N, )
        Time in seconds.
    angles : array_like, shape(N, 3)
        hip flexion, knee extension, ankle dorsiflexion
    torques : array_like, shape(N, 3)
        hip flexion, knee extension, ankle dorsiflexion
    angles_meas : array_like, shape(N, 3)
        hip flexion, knee extension, ankle dorsiflexion
    torques_meas : array_like, shape(N, 3), optional
        hip flexion, knee extension, ankle dorsiflexion
    grf : array_like, shape(N, 2), optional
        horizontal, vertical
    grf_meas : array_like, shape(N, 2), optional
        horizontal, vertical

    Returns
    =======
    axes : shape(2,) or shape(3,)
        First axis contains the angles, second axis contains the joint torques,
        third axis contains the ground reaction forces.

    """
    if grf is not None:
        fig, axes = plt.subplots(3, 1, figsize=(6.0, 9.0))
        grf_labels = ('horizontal', 'vertical')
    else:
        fig, axes = plt.subplots(2, 1, figsize=(6.0, 9.0))
    colors = ('C0', 'C1', 'C2')

    anglabels = ('hip flexion', 'knee extension', 'ankle dorsiflexion')
    for ang, ang_meas, color, lab in zip(angles.T, angles_meas.T, colors,
                                         anglabels):
        axes[0].plot(t, ang, color=color, label=lab)
        axes[0].plot(t, ang_meas, color=color, linestyle='--',
                     label=lab + ' measured')
    axes[0].legend()
    axes[0].set_ylabel('Angle [deg]')

    torlabels = ('hip flexion', 'knee extension', 'ankle dorsiflexion')
    for tor, color, lab in zip(torques.T, colors, torlabels):
        axes[1].plot(t, tor, color=color, label=lab)
    if torques_meas is not None:
        for tor, color, lab in zip(torques_meas.T, colors, torlabels):
            axes[1].plot(t, tor, color=color, linestyle='--',
                         label=lab + ' measured')
    axes[1].legend()
    axes[1].set_ylabel('Torque [Nm]')
    axes[1].set_xlabel('Time [s]')

    if grf is not None:
        for grf_com, color, lab in zip(grf.T, colors, grf_labels):
            axes[2].plot(t, grf_com, color=color, label=lab)
    if grf_meas is not None:
        for grf_com, color, lab in zip(grf_meas.T, colors, grf_labels):
            axes[2].plot(t, grf_com, color=color, linestyle='--',
                         label=lab + ' measured')

    if (grf is not None) or (grf_meas is not None):
        axes[2].set_ylabel('Ground reaction force [N]')
        axes[2].set_xlabel('Time [s]')
        axes[2].legend()

    return axes


def body_segment_parameters_from_calibration(calibration_csv_path,
                                             subject_mass, plot=False):
    """Generates model segment dimensions, mass, mass center dimensions, and
    central moments of inertia based on the calibration pose marker set and the
    subject's total mass using Winter's body segment scaling table.

    Parameters
    ==========
    calibration_csv_path : str
        Path to a file containing the time series of the markers during a
        calibration pose (subject is stationary).
    subject_mass: float
        Total mass of the subject.
    plot : boolean, optional
        If true a plot of the markers in the mean position will be shown.

    Returns
    =======
    constants : dictionary
        Mapping of model parameter (segment and mass center dimensions, central
        moment of inertia, mass) string name to float.

    """

    df = pd.read_csv(calibration_csv_path)

    if plot:
        # x: positive heel to toe
        # y: positive foot to head
        # z: positive left to right

        df_mkrs = df[df.columns[df.columns.str.endswith('PosX') |
                                df.columns.str.endswith('PosY') |
                                df.columns.str.endswith('PosZ')]]

        x = df_mkrs[df_mkrs.columns[df_mkrs.columns.str.endswith('PosX')]]
        y = df_mkrs[df_mkrs.columns[df_mkrs.columns.str.endswith('PosY')]]
        z = df_mkrs[df_mkrs.columns[df_mkrs.columns.str.endswith('PosZ')]]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x.mean(), z.mean(), y.mean())
        xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, num=10),
                             np.linspace(-0.5, 0.5, num=10))
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.5, color='black')
        ax.invert_yaxis()
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_aspect('equal')
        plt.show()

    def length(marker_one, marker_two, project=None):
        """Returns the Euclidean distances between two markers versus time.

        Parameters
        ==========
        marker_one : string
            Full marker name, e.g. 'RHEE'.
        marker_two : string
            Full marker name, e.g. 'RHEE'.
        project: str, optional
            Project the markers onto the plane normal to the provided axis
            label, i.e. 'x' (coronal plane), 'y' (transverse plane), or 'z'
            (sagittal plane).

        """
        x1 = df[marker_one + '.PosX']
        y1 = df[marker_one + '.PosY']
        z1 = df[marker_one + '.PosZ']

        x2 = df[marker_two + '.PosX']
        y2 = df[marker_two + '.PosY']
        z2 = df[marker_two + '.PosZ']

        if project == 'x':
            sum_of_squares = (y2 - y1)**2 + (z2 - z1)**2
        elif project == 'y':
            sum_of_squares = (x2 - x1)**2 + (z2 - z1)**2
        elif project == 'z':
            sum_of_squares = (x2 - x1)**2 + (y2 - y1)**2
        elif project is None:
            sum_of_squares = (x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2

        return np.sqrt(sum_of_squares)

    def mean_length(marker_one, marker_two, project=None):
        """Returns the length between markers as the mean of right and left.
        Provide marker names sans the 'R' or 'L' indicator, i.e. 'HEE' not
        'RHEE'."""
        return np.mean((
            length('R' + marker_one, 'R' + marker_two,
                   project=project).mean(),  # right
            length('L' + marker_one, 'L' + marker_two,
                   project=project).mean(),  # left
        ))

    # Markers in our set:
    # Shoulder, SHO, acromion marker is 35 mm above the glenohumeral joint
    # Greater trochanter, GTRO
    # Lateral epicondyle of knee, LEK
    # Lateral malleolus, LM
    # Heel (placed at same height as marker 6), HEE
    # Head of 5th metatarsal, MT5
    # Tip of big toe, TOE

    # Location of glenohumeral joint is 35 mm below the acromion (De Leva, J
    # Biomech 1996)
    len_trunk = mean_length('SHO', 'GTRO', project='z') - 0.035
    len_thigh = mean_length('GTRO', 'LEK', project='z')
    len_shank = mean_length('LEK', 'LM', project='z')
    len_foot = mean_length('HEE', 'TOE', project='z')

    def foot_dimensions():
        hxd = -(df['RLM.PosX'] - df['RHEE.PosX']).mean()  # - marker_diameter/2
        txd = (df['RMT5.PosX'] - df['RLM.PosX']).mean()
        fyd = -df['RLM.PosY'].mean()
        xd = 0.5*len_foot + hxd
        yd = 0.5*fyd

        hxg = -(df['LLM.PosX'] - df['LHEE.PosX']).mean()  # - marker_diameter/2
        txg = (df['LMT5.PosX'] - df['LLM.PosX']).mean()
        fyg = -df['LLM.PosY'].mean()
        xg = 0.5*len_foot + hxg
        yg = 0.5*fyg

        return ((xg + xd)/2, (yg + yd)/2, (hxg + hxd)/2, (txg + txd)/2,
                (fyg + fyd)/2)

    # Winter Table 4.1 selected rows:
    # Segment name, segment landmarks, percent mass
    # Foot, Lateral malleolus/head metatarsal II, 0.0145
    # Leg, Femoral condyles/medial malleolus, 0.433
    # Thigh, Greater trochanter/femoral condyles, 0.1
    # Head, arms, and trunk (HAT), Greater trochater/glenohumeral joint*, 0.678
    mass_trunk = 0.678*subject_mass
    mass_thigh = 0.1*subject_mass
    mass_shank = 0.0465*subject_mass
    mass_foot = 0.0145*subject_mass

    # Make sure mass totals to subject's total mass.
    np.testing.assert_allclose(
        subject_mass,
        mass_trunk + 2*mass_thigh + 2*mass_shank + 2*mass_foot
    )

    x, y, hx, tx, fy = foot_dimensions()

    constants = {
        # trunk, a
        'ma': mass_trunk,
        'ia': mass_trunk*(0.496*len_trunk)**2,
        'xa': 0.0,
        'ya': 0.626*len_trunk,  # TODO: distal or proximal?
        # rthigh, b
        'mb': mass_thigh,
        'ib': mass_thigh*(0.323*len_thigh)**2,
        'xb': 0.0,
        'yb': -0.433*len_thigh,
        'lb': len_thigh,
        # rshank, c
        'mc': mass_shank,
        'ic': mass_shank*(0.302*len_shank)**2,
        'xc': 0.0,
        'yc': -0.433*len_shank,
        'lc': len_shank,
        # rfoot, d
        'md': mass_foot,
        'id': mass_foot*(0.475*len_foot)**2,
        'xd': x,
        'yd': y,
        'hxd': hx,
        'txd': tx,
        'fyd': fy,
        # lthigh, e
        'me': mass_thigh,
        'ie': mass_thigh*(0.323*len_thigh)**2,
        'xe': 0.0,
        'ye': -0.433*len_thigh,
        'le': len_thigh,
        # lshank, f
        'mf': mass_shank,
        'if': mass_shank*(0.302*len_shank)**2,
        'xf': 0.0,
        'yf': -0.433*len_shank,
        'lf': len_shank,
        # lfoot, g
        'mg': mass_foot,
        'ig': mass_foot*(0.475*len_foot)**2,
        'xg': x,
        'yg': y,
        'hxg': hx,
        'txg': tx,
        'fyg': fy,
    }

    return constants


def plot_marker_comparison(marker_coords, marker_labels, marker_df, prob,
                           solution):
    """Returns plot comparing measured marker locations to the model's
    motion.

    Parameters
    ==========
    marker_coords : iterable of symbols
        In groups of 4: [m1_lx, m1_ly, m1_rx, m1_ry, ...]
    marker_labels : iterable of strings
        In groups of 4: [LM1.PosX, LM1.PosY, RM1.PosX, RM1.PosY, ...]
    marker_df : DataFrame
        Has columns for each marker label. Rows are time instances.
    prob : Problem
        Fully defined Prolbem.
    solution : ndarray, shape(n, )
        Solution generated by Problem.

    """
    fig, ax = plt.subplots()

    # NOTE : assumes pairs of markers for left and right
    for i in range(len(marker_coords)//4):
        lx, ly, rx, ry = marker_coords[i*4:(i + 1)*4]
        lx_lab, ly_lab, rx_lab, ry_lab = marker_labels[i*4:(i + 1)*4]

        ax.plot(prob.extract_values(solution, lx),
                prob.extract_values(solution, ly),
                color=f'C{i}',
                linestyle='-',
                label=f'{lx_lab}, Model')

        ax.plot(marker_df[lx_lab], marker_df[ly_lab],
                color=f'C{i}',
                linestyle='--',
                label=f'{lx_lab}, Data')

        ax.plot(prob.extract_values(solution, rx),
                prob.extract_values(solution, ry),
                color=f'C{i}',
                linestyle='-',
                label=f'{rx_lab}, Model')

        ax.plot(marker_df[rx_lab], marker_df[ry_lab],
                color=f'C{i}',
                linestyle=':',
                label=f'{rx_lab}, Data')

    ax.set_aspect("equal")
    ax.legend()

    return ax


if __name__ == "__main__":
    constants = body_segment_parameters_from_calibration(CALIBDATAPATH, 70.0,
                                                         plot=True)
    master_df = pd.read_csv(GAITDATAPATH)
    df = extract_gait_cycle(master_df, 100)
    plot_points(df)

    half_cycle_num_nodes = 50
    full_cycle_num_nodes = 121

    # extracts a gait cycle from our PeerJ data
    (duration, walking_speed, num_angles, ang_data, marker_df,
     kinetic_df, ang_df, _, sample_data_percent) = load_sample_data(
         half_cycle_num_nodes, gait_cycle_number=6)
    kinetic_df.plot(marker='.', subplots=True,
                    title="Kinetics from PeerJ Data")

    # show that the full gait cycle generates correctly
    winter_full_df = load_winter_data_frame(num_nodes=full_cycle_num_nodes)
    winter_full_df.plot(x='Percent Gait Cycle', marker='.', subplots=True,
                        layout=(-1, 2), title="Winters' Data as Full Cycle")

    # this creates the same output as load_winder_data()
    winter_df = load_winter_data_frame(num_nodes=half_cycle_num_nodes,
                                       half_cycle=True, drop_last_node=True)

    # this loads Winters' data as per Ton's original implemetnation
    _, _, num_ang, ang_data = load_winter_data(half_cycle_num_nodes,
                                               as_data_frame=True)

    things = [
        'FP2.ForX',  # right
        'FP1.ForX',  # left

        'FP2.ForY',
        'FP1.ForY',

        'Right.Hip.Flexion.Angle',
        'Left.Hip.Flexion.Angle',

        'Right.Knee.Extension.Angle',
        'Left.Knee.Extension.Angle',

        'Right.Ankle.DorsiFlexion.Angle',
        'Left.Ankle.DorsiFlexion.Angle',

        'Right.Hip.Flexion.Moment',
        'Left.Hip.Flexion.Moment',

        'Right.Knee.Extension.Moment',
        'Left.Knee.Extension.Moment',

        'Right.Ankle.DorsiFlexion.Moment',
        'Left.Ankle.DorsiFlexion.Moment',
    ]
    fig, axes = plt.subplots(len(things)//2, 2,
                             sharex=True,
                             sharey='row',
                             layout='constrained')
    for ax, col in zip(axes.flatten(), things):
        if col in ang_data:
            if 'Angle' in col:
                val = np.rad2deg(ang_data[col])
            else:
                val = ang_data[col]
            ax.plot(ang_data['Percent Gait Cycle'], val, color='C0',
                    marker='o', label='Winter (original): ' + col)
        if col in winter_df:
            ax.plot(winter_df['Percent Gait Cycle'], winter_df[col],
                    color='C1', marker='.', label='Winter (new): ' + col)
        if col in kinetic_df:
            ax.plot(sample_data_percent, kinetic_df[col], marker='.',
                    color='C2', label='Measured: ' + col)
        if col in ang_df:
            ax.plot(sample_data_percent, np.rad2deg(ang_df[col]), color='C2',
                    marker='.', label='Measured: ' + col)
        ax.axvline(50.0, color='black')  # 50%
        ax.legend(fontsize=6)

    axes[0, 0].set_title('Right')
    axes[0, 1].set_title('Left')
    axes[-1, 0].set_xlabel('Percent Gait Cycle')
    axes[-1, 1].set_xlabel('Percent Gait Cycle')

    plt.show()
