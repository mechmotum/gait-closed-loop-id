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
import matplotlib.pyplot as plt

DATAFILE = '020-longitudinal-perturbation-gait-cycles.csv'
DATAPATH = os.path.join(os.path.dirname(__file__), '..', 'data', DATAFILE)


def extract_gait_cycle(df, number):

    if number not in df['major'].values:
        msg = '{} not in {}-{}'
        raise ValueError(msg.format(number, df['major'].min(),
                                    df['major'].max()))

    return df[df['major'] == number]


def plot_points(df):

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


if __name__ == "__main__":
    master_df = pd.read_csv(DATAPATH)
    df = extract_gait_cycle(master_df, 100)
    plot_points(df)
    plt.show()
