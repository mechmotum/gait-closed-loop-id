"""Finds a minimum-effort standing state using the same model as the gait
solution. We do this as a trajectory optimization with two time points to
create an initial guess for the walking solution.

"""

import os

from opty import Problem
from pygait2d import simulate
from pygait2d.derive import derive_equations_of_motion
from pygait2d.segment import time_symbol
import numpy as np

from utils import DATADIR, get_sym_by_name


def find_standing_state():
    """Returns a single node of the solution vector for a standing state and
    saves it to DATADIR/standing.csv."""

    num_nodes = 2
    h = 1.0  # time step does not matter, since it is a static equilibrium

    symbolics = derive_equations_of_motion(
        prevent_ground_penetration=False,
        treadmill=True,
        hand_of_god=False
    )

    qax, qay, qa, qb, qc, qd, qe, qf, qg = symbolics.coordinates
    uax, uay, ua, ub, uc, ud, ue, uf, ug = symbolics.speeds
    v = get_sym_by_name(symbolics.specifieds, 'v')

    par_map = simulate.load_constants(symbolics.constants,
                                      os.path.join(DATADIR,
                                                   'example_constants.yml'))

    # Set belt velocity v(t) to zero.
    traj_map = {
        v: np.zeros(num_nodes),
    }

    # Bound all the states
    bounds = {
        qay: (0.5, 1.5),
    }
    # trunk, hip, knee, ankle angles should all be small
    bounds.update({k: (-0.1, 0.1) for k in [qa, qb, qc, qd, qe, qf, qg]})
    # horizontal position and all speeds should be zero
    bounds.update({k: (0.0, 0.0)
                   for k in [qax, uax, uay, ua, ub, uc, ud, ue, uf, ug]})
    # joint torques should be small
    bounds.update({k: (-10.0, 10.0) for k in symbolics.joint_torques})

    def obj(p, free):
        """Return sum of squares of joint torques."""
        tor_vals = p.extract_values(free, *symbolics.joint_torques)
        return np.sum(tor_vals**2)

    def obj_grad(p, free):
        grad = np.zeros_like(free)
        tor_vals = p.extract_values(free, *symbolics.joint_torques)
        p.fill_free(grad, 2.0*tor_vals, *symbolics.joint_torques)
        return grad

    prob = Problem(
        obj,
        obj_grad,
        symbolics.equations_of_motion,
        symbolics.states,
        num_nodes,
        h,
        known_parameter_map=par_map,
        known_trajectory_map=traj_map,
        bounds=bounds,
        time_symbol=time_symbol,
    )

    # Find the optimal solution and save it if it converges.
    initial_guess = (0.5*(prob.lower_bound + prob.upper_bound) +
                     0.01*(prob.upper_bound - prob.lower_bound)*
                     np.random.random_sample(prob.num_free))
    solution, info = prob.solve(initial_guess)

    # Only keep the first node (the second is the same).
    solution = solution[0:-1:2]
    if info['status'] == 0:
        np.savetxt(os.path.join(DATADIR, 'standing.csv'), solution,
                   fmt='%.15f')
        return solution
    else:
        RuntimeError('Ipopt did not converge to a solution.')


if __name__ == "__main__":
    standing_sol = find_standing_state()
    print(standing_sol)
