"""
Weirdly, this does not match the coordinate system on the treadmill in Figure 1
in my paper. There it shows the direction of travel in the -Z direction. Maybe
GTK transforms the axes and the raw data is in the Cortex axes.
X : positive forwards
Y : postive up

major (gait cycle number)
minor (percent gait cycle)
Original Time
LeftBeltSpeed
RightBeltSpeed

.PosX
.PosY

LSHO: left shoulder
RSHO

LGTRO: Left greater trochanter of the femur
RGTRO

LLEK: Left lateral epicondyle of the knee
LLM: Left lateral malleoulus of the ankle
LHEE
LTOE

RLEK
RLM
RHEE
RTOE


4: 192. Left.Hip.Flexion.Angle
5: 193. Left.Knee.Flexion.Angle
6: 194. Left.Ankle.PlantarFlexion.Angle
7: 195. Left.Hip.Flexion.Rate
8: 196. Left.Knee.Flexion.Rate
9: 197. Left.Ankle.PlantarFlexion.Rate
10: 207. Right.Hip.Flexion.Angle
11: 208. Right.Knee.Flexion.Angle
12: 209. Right.Ankle.PlantarFlexion.Angle
13: 210. Right.Hip.Flexion.Rate
14: 211. Right.Knee.Flexion.Rate
15: 212. Right.Ankle.PlantarFlexion.Rate

"""
import os

import numpy as np
import pandas as pd
import sympy as sm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pygait2d.segment import time_varying, contact_force
from symmeplot.matplotlib import Scene3D
from matplotlib.animation import FuncAnimation

GAITFILE = '020-longitudinal-perturbation-gait-cycles.csv'
CALIBFILE = '020-calibration-pose.csv'
DATADIR = os.path.join(os.path.dirname(__file__), '..', 'data')
GAITDATAPATH = os.path.join(DATADIR, GAITFILE)
CALIBDATAPATH = os.path.join(DATADIR, CALIBFILE)


def animate(symbolics, xs, rs, h, speed, times, par_map):

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
    v = symbolics.specifieds[-1]
    scene.add_vector(contact_force(rfoot.toe, ground, origin, v)/600.0,
                     rfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(rfoot.heel, ground, origin, v)/600.0,
                     rfoot.heel, color="tab:blue")
    scene.add_vector(contact_force(lfoot.toe, ground, origin, v)/600.0,
                     lfoot.toe, color="tab:blue")
    scene.add_vector(contact_force(lfoot.heel, ground, origin, v)/600.0,
                     lfoot.heel, color="tab:blue")

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
        (contact_force(rfoot.toe, ground, origin, v) +
            contact_force(rfoot.heel, ground, origin, v)).to_matrix(ground),
        cse=True)

    eval_lforce = sm.lambdify(
        symbolics.states + symbolics.specifieds + symbolics.constants,
        (contact_force(lfoot.toe, ground, origin, v) +
            contact_force(lfoot.heel, ground, origin, v)).to_matrix(ground),
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

    # TODO : Only tracking ankle, need to scale model before tracking multiple
    # markers works.
    points = {
        'ank_l': lshank.joint,  # left ankle
        'ank_r': rshank.joint,  # right ankle
        #'hel_l': lfoot.heel,
        #'hel_r': rfoot.heel,
        #'hip_l': trunk.joint,  # hip
        #'hip_r': trunk.joint,  # hip
        #'kne_l': lthigh.joint,  # left knee
        #'kne_r': rthigh.joint,  # right knee
        #'toe_l': lfoot.toe,
        #'toe_r': rfoot.toe,
    }

    point_data_map = {
        'ank_l': 'LLM',
        'ank_r': 'RLM',
        #'hel_l': 'LHEE',
        #'hel_r': 'RHEE',
        #'hip_l': 'LGTRO',
        #'hip_r': 'RGTRO',
        #'kne_l': 'LLEK',
        #'kne_r': 'RLEK',
        #'toe_l': 'LTOE',
        #'toe_r': 'RTOE',
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


def load_winter_data(num_nodes):
    """Returns interpolated normative gait data from Winter's book.

    Returns
    =======
    duration : float
        Time in seconds of the half gait cycle.
    walking_speed : float
        Average walking speed in meters per second.
    num_angles : int
        Numer of angles: 6. (r & l hip, knee, ankle)
    ang_data : ndarray, shape((num_nodes-1)*num_angles,)
        Angle data in radians linear interpolated at the times corresponding to
        the number of nodes. [hip1, ..., hipN, knee1, ..., kneeN, ankle1, ...,
        ankleN, hip1, ..., hipN, knee1, ..., kneeN, ankle1, ..., ankleN]

    """
    fname = os.path.join(os.path.dirname(__file__),
                         os.path.join('..', 'data', 'Winter_normal.csv'))

    data = np.genfromtxt(fname, delimiter=',')

    # extract gait cycle duration and speed
    duration = data[1, 2]/2  # half gait cycle duration
    walking_speed = data[2, 2]

    # extract hip, knee, ankle angle (full gait cycle)
    ang = np.deg2rad(data[6:57, 4:7])
    # invert Winter's knee angle, to be compatible with our model
    ang[:, 1] = -ang[:, 1]

    # convert full gait cycle (one side) into a half gait cycle for both sides
    # and resample to num_nodes
    ang = np.concatenate((ang[:26, :], ang[25:, :]), axis=1)
    rows, num_angles = ang.shape
    ang_resampled = np.zeros((num_nodes - 1, num_angles))
    t = np.arange(0, rows)/(rows - 1)  # gait phase from data
    t_new = np.arange(0, num_nodes - 1)/(num_nodes - 1)  # gait phase for sim
    for i in range(0, num_angles):
        ang_resampled[:, i] = np.interp(t_new, t, ang[:, i])

    # ang_resampled shape(time, [hip, knee, ankle, hip, knee, ankle])

    # store the angle trajectories in a 1d array, for tracking
    ang_data = ang_resampled.transpose().flatten()

    return duration, walking_speed, num_angles, ang_data


def load_sample_data(num_nodes, gait_cycle_number=100):
    """Returns data from a measurement file in the same format as
    load_winter_data.

    Parameters
    ==========
    num_nodes : integer
        Number of evenly spaced time instances in the time series.
    gait_cycle_number: integer, optional
        Number from 0 to total gait cycles - 1.

    """
    master_df = pd.read_csv(GAITDATAPATH)
    df = extract_gait_cycle(master_df, gait_cycle_number)

    df = df.iloc[:11, :]  # take 0% to 50%

    time = df['Original Time'].values
    time -= time[0]
    duration = time[-1] - time[0]  # 0% to 50% duration

    walking_speed = 1.2  # nominal speed from trial 20 meta data

    angles = [
        'Right.Hip.Flexion.Angle',
        'Right.Knee.Flexion.Angle',
        'Right.Ankle.PlantarFlexion.Angle',
        'Left.Hip.Flexion.Angle',
        'Left.Knee.Flexion.Angle',
        'Left.Ankle.PlantarFlexion.Angle',
    ]
    ang_arr = -df[angles].values  # change to extension (knee and ankle)

    ang_arr[:, [0, 3]] *= -1  # change hip back to flexion
    ang_arr[:, [2, 5]] -= np.pi/2

    markers = [
        'LGTRO.PosX',
        'LGTRO.PosY',
        'LHEE.PosX',
        'LHEE.PosY',
        'LLEK.PosX',
        'LLEK.PosY',
        'LLM.PosX',
        'LLM.PosY',
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
        'RSHO.PosX',
        'RSHO.PosY',
        'RTOE.PosX',
        'RTOE.PosY',
    ]
    marker_vals = df[markers].values

    kinetics = [
        #'FP1.ForX',
        #'FP1.ForY',
        #'FP1.ForZ',
        #'FP2.ForX',
        #'FP2.ForY',
        #'FP2.ForZ',
        'Left.Hip.Flexion.Moment',
        'Left.Knee.Flexion.Moment',
        'Left.Ankle.PlantarFlexion.Moment',
        'Right.Hip.Flexion.Moment',
        'Right.Knee.Flexion.Moment',
        'Right.Ankle.PlantarFlexion.Moment',
    ]
    kinetic_vals = df[kinetics].values

    new_time = np.linspace(0.0, duration, num=num_nodes - 1)
    interp_ang_arr = interp1d(time, ang_arr, axis=0)(new_time)
    interp_mark_arr = interp1d(time, marker_vals, axis=0)(new_time)
    interp_kinetic_arr = interp1d(time, kinetic_vals, axis=0)(new_time)

    mark_df = pd.DataFrame(dict(zip(markers, interp_mark_arr.T)))
    kinetic_df = pd.DataFrame(dict(zip(kinetics, interp_kinetic_arr.T)))

    return (duration, walking_speed, len(angles), interp_ang_arr.T.flatten(),
            mark_df, kinetic_df)


def plot_joint_comparison(t, angles, torques, angles_meas, torques_meas=None):
    """
    Parameters
    ==========
    t : array_like, shape(N, )
        Time in seconds.
    angles : array_like, shape(N, 3)
        hip flexion, knee flexion, ankle dorsiflexion
    torques : array_like, shape(N, 3)
        hip extension, knee extension, ankle plantarflexion
    angles_meas : array_like, shape(N, 3)
        hip flexion, knee flexion, ankle dorsiflexion
    torques_meas : array_like, shape(N, 3), optional
        hip extension, knee extension, ankle plantarflexion

    Returns
    =======
    axes : shape(2,)

    """
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 9.0))
    colors = ('C0', 'C1', 'C2')

    anglabels = ('hip flexion', 'knee flexion', 'ankle dorsiflexion')
    for ang, ang_meas, color, lab in zip(angles.T, angles_meas.T, colors,
                                         anglabels):
        axes[0].plot(t, ang, color=color, label=lab)
        axes[0].plot(t, ang_meas, color=color, linestyle='--',
                     label=lab + ' measured')
    axes[0].legend()
    axes[0].set_ylabel('Angle [deg]')

    torlabels = ('hip extension', 'knee extension', 'ankle plantarflexion')
    for tor, color, lab in zip(torques.T, colors, torlabels):
        axes[1].plot(t, tor, color=color, label=lab)
    if torques_meas is not None:
        for tor, color, lab in zip(torques_meas.T, colors, torlabels):
            axes[1].plot(t, tor, color=color, label=lab + ' measured')
    axes[1].legend()
    axes[1].set_ylabel('Torque [Nm]')
    axes[1].set_xlabel('Time [s]')

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


if __name__ == "__main__":
    constants = body_segment_parameters_from_calibration(CALIBDATAPATH, 70.0,
                                                         plot=True)
    master_df = pd.read_csv(GAITDATAPATH)
    df = extract_gait_cycle(master_df, 100)
    plot_points(df)
    plt.show()
