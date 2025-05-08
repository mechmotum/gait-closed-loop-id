#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external libraries
import sympy as sm
import sympy.physics.mechanics as me
from pygait2d.segment import (BodySegment, TrunkSegment, FootSegment,
                              time_varying, time_symbol, sym_kwargs)

me.dynamicsymbols._t = time_symbol


def contact_force(point, ground, origin, v_belt):
    """Returns a contact force vector acting on the given point made of
    friction along the contact surface and elastic force in the vertical
    direction.

    Parameters
    ==========
    point : sympy.physics.mechanics.Point
        The point which the contact force should be computed for.
    ground : sympy.physics.mechanics.ReferenceFrame
        A reference frame which represents the inerital ground in 2D space.
        The x axis defines the ground line and positive y is up.
    origin : sympy.physics.mechanics.Point
        An origin point located on the ground line.

    Returns
    =======
    force : sympy.physics.mechanics.Vector
        The contact force between the point and the ground.

    """
    # This is the "height" of the point above the ground, where a negative
    # value means that the point is below the ground.
    y_location = point.pos_from(origin).dot(ground.y)

    # The penetration into the ground is mathematically defined as:
    #
    #               { 0 if y_location > 0
    # deformation = {
    #               { abs(y_location) if y_location < 0
    #

    penetration = (sm.Abs(y_location) - y_location) / 2

    velocity = point.vel(ground)

    # The addition of "- y_location" here adds a small linear term to the
    # cubic stiffness and creates a light attractive force torwards the
    # ground. This is in place to ensure that gradients can be computed for
    # the optimization used in Ackermann and van den Bogert 2010.
    contact_stiffness, contact_damping = sm.symbols('kc, cc', **sym_kwargs)
    contact_friction_coefficient, friction_scaling_factor = \
        sm.symbols('mu, vs', **sym_kwargs)

    vertical_force = (contact_stiffness * penetration ** 3 - y_location) * \
        (1 - contact_damping * velocity.dot(ground.y))

    # friction force depends on velocity of the contact point relative
    # to the treadmill belt (which is moving backwards)
    friction = -contact_friction_coefficient * vertical_force * \
        ((2 / (1 + sm.exp( (-v_belt-velocity.dot(ground.x)) /
                          friction_scaling_factor))) - 1)

    return friction * ground.x + vertical_force * ground.y


def derive_equations_of_motion(gait_cycle_control=False):
    """Returns the equations of motion for the planar walking model along with
    all of the constants, coordinates, speeds, joint torques, visualization
    frames, inertial reference frame, and origin point.

    Parameters
    ==========
    gait_cycle_control : boolean, optinal, default=False
        If true, the specified forces and torques are replaced with a full
        state feeback controller summed with the forces and torques.

    Returns
    =======
    mass_matrix : Matrix, shape(18, 18)
        Full mass matrix of the system to be multiplied by ``x' =
        [coordinates', speeds']``.
    forcing_vector : Matrix, shape(18, 1)
        Full forcing vector where: ``mass_matrix*x' = forcing vector``.
    kane : sympy.physics.mechanics.Kane
        A KanesMethod object in which the equations of motion have been
        derived. All symbolics are accessible from this object if needed.
    constants : list of Symbol
        The constants in the equations of motion.
    coordinates : list of Function(t)
        The generalized coordinates of the system.
    speeds : list of Function(t)
        The generalized speeds of the system.
    specified : list of Function(t), optional, default=None
        The specifed quantities of the system.
    visualization_frames : list of VizFrame
    ground : ReferenceFrame
        An inertial reference frame representing the Earth and a the direction
        of the uniform gravitational field.
    origin : Point
        A point fixed in the ground reference frame used for calculating
        translational velocities.
    segments : list of Segment
        All of the segment objects that make up the human.

    """

    print('Forming positions, velocities, accelerations and forces.')
    segment_descriptions = {'A': (TrunkSegment, 'Trunk', 'Hip'),
                            'B': (BodySegment, 'Right Thigh', 'Right Knee'),
                            'C': (BodySegment, 'Right Shank', 'Right Ankle'),
                            'D': (FootSegment, 'Right Foot', 'Right Heel'),
                            'E': (BodySegment, 'Left Thigh', 'Left Knee'),
                            'F': (BodySegment, 'Left Shank', 'Left Ankle'),
                            'G': (FootSegment, 'Left Foot', 'Left Heel')}

    # define a symbol for the belt velocity
    v = time_varying('v')
    
    ground = me.ReferenceFrame('N')
    origin = me.Point('O')
    origin.set_vel(ground, 0)

    segments = []
    constants = []
    coordinates = []
    speeds = []
    specified = [v]
    kinematic_equations = []
    external_forces_torques = []
    bodies = []
    visualization_frames = []

    for label in sorted(segment_descriptions.keys()):

        segment_class, desc, joint_desc = segment_descriptions[label]

        if label == 'A':  # trunk
            parent_reference_frame = ground
            origin_joint = origin
        elif label == 'E':  # left thigh
            # For the left thigh, set the trunk and hip as the
            # reference_frame and origin joint.
            parent_reference_frame = segments[0].reference_frame
            origin_joint = segments[0].joint
        else:  # thighs, shanks
            parent_reference_frame = segments[-1].reference_frame
            origin_joint = segments[-1].joint

        segment = segment_class(label, desc, parent_reference_frame,
                                origin_joint, joint_desc, ground)
        segments.append(segment)

        # constants, coordinates, speeds, kinematic differential equations
        if label == 'A':  # trunk
            coordinates += segment.qa
            speeds += segment.ua
            constants += segment.constants
        else:
            # skip g for all segments but the trunk
            constants += segment.constants[1:]

        coordinates.append(segment.generalized_coordinate_symbol)
        speeds.append(segment.generalized_speed_symbol)

        kinematic_equations += segment.kinematic_equations

        # gravity
        external_forces_torques.append((segment.mass_center,
                                        segment.gravity))

        # joint torques
        external_forces_torques.append((segment.reference_frame,
                                        segment.torque))
        external_forces_torques.append((segment.parent_reference_frame,
                                        -segment.torque))
        specified.append(segment.joint_torque_symbol)

        # contact force
        if label == 'D' or label == 'G':  # foot
            external_forces_torques.append((segment.heel,
                                            contact_force(segment.heel,
                                                          ground, origin, v)))
            external_forces_torques.append((segment.toe,
                                            contact_force(segment.toe,
                                                          ground, origin, v)))

        # bodies
        bodies.append(segment.rigid_body)

        visualization_frames += segment.visualization_frames()

    # add contact model constants
    # TODO : these should be grabbed from the segments, not recreated.
    constants += list(sm.symbols('kc, cc, mu, vs', real=True, positive=True))

    # equations of motion
    print("Initializing Kane's Method.")
    kane = me.KanesMethod(ground, coordinates, speeds, kinematic_equations)
    print("Forming Kane's Equations.")
    kane.kanes_equations(bodies, loads=external_forces_torques)
    mass_matrix = kane.mass_matrix_full
    forcing_vector = kane.forcing_full

    if gait_cycle_control:
        # joint_torques(phase) = mean_joint_torque + K*(joint_state_desired -
        # joint_state)
        # r = [Fax(t), Fay(t), Ta(t), Tb(t), Tc(t), Td(t), Te(t), Tf(t), Tg(t)]
        # x = [qax(t), qay(t), qa(t), qb(t), qc(t), qd(t), qe(t), qf(t), qg(t),
        #      uax(t), uay(t), ua(t), ub(t), uc(t), ud(t), ue(t), uf(t), ug(t)]
        # commanded states
        # xc = [qax_c(t), qay_c(t), qa_c(t), qb_c(t), qc_c(t), qd_c(t), qe_c(t), qf_c(t), qg_c(t)]
        #       uax_c(t), uay_c(t), ua_c(t), ub_c(t), uc_c(t), ud_c(t), ue_c(t), uf_c(t), ug_c(t)]
        # controlled joint torques
        # uc(t) = r(t) + K(t)*(xc(t) - x(t))
        # r(t) : force or torque
        # K(t) : time varying full state feedback gain matrix
        # xc(t) : commanded (desired) states
        # x(t) : states
        # K is, in general, 9 x 18
        # the first three rows and columns will be zero if hand of god is
        # absent, which effectively makes it a 6x18
        # K = |kax_qax, kax_qay, kax_qa, kax_qb, kax_qc, kax_qd, kax_qe, kax_qf, kax_qg,
        #      kax_uax, kax_uay, kax_ua, kax_ub, kax_uc, kax_ud, kax_ue, kax_uf, kax_ug|
        #     |kay_qax, kay_qay, kay_qa, kay_qb, kay_qc, kay_qd, kay_qe, kay_qf, kay_qg|
        #     |ka_qax, ka_qay, ka_qa, ka_qb, ka_qc, ka_qd, ka_qe, ka_qf, ka_qg|
        #     ...
        #     |kg_qax, kg_qay, kg_qa, kg_qb, kg_qc, kg_qd, kg_qe, kg_qf, kg_qg|
        # We can just go through the final equations of motion and replace the
        # joint torques Tb through Tg with Tb -> Tb + kb_qb*(qb_des - qb) +
        # kb_ub*(ub_des - qb) + ...
        K = []
        for ri in specified:
            row = []
            for xi in coordinates + speeds:
                row.append(sm.Function('k_{}_{}'.format(ri.name, xi.name),
                                       real=True)(time_symbol))
            K.append(row)
        K = sm.Matrix(K)

        xc = []
        for xi in coordinates + speeds:
            xc.append(sm.Function('{}_c'.format(xi.name),
                                  real=True)(time_symbol))
        r = sm.Matrix(specified)
        xc = sm.Matrix(xc)
        x = sm.Matrix(coordinates + speeds)

        uc = r + K@(xc - x)

        repl = {k: v for k, v in zip(specified, uc)}

        forcing_vector = forcing_vector.xreplace(repl)

        specified += K[:]
        specified += xc[:]

    return (mass_matrix, forcing_vector, kane, constants, coordinates, speeds,
            specified, visualization_frames, ground, origin, segments)
