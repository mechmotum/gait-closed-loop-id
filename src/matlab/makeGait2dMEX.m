function model = makeGait2dMEX()
% This function produces a MEX function gait2dMEX for the inverse
% multibody dynamics of the Gait2d model, computing the generalized forces Q that
% are required for the model to be in kinematic state q,qd,qdd and belt
% speed is v.
% We also calculate the Jacobians of Q with respect to q,qd,qdd so we
% can do efficient optimization or use Newton's method for simulation.
%
% MEX function usage: [Q,dQdq,dQdqd,dQdqdd,FK,GRF] = gait2dMEX(q,qd,qdd,v)
%
% TODO: use "contactBuiltin=false" to produce a MEX function with
% GRF input, so that the contact model can be hand-coded in Matlab,
% with additional contact states x,y,Fx,Fy for each contact point

    % process the options
    if (nargin < 1)
        options = struct();
    end
    if ~isfield(options,'contactBuiltin') options.contactBuiltin = true; end
    
    % MEX function binary we create
    mexbinary = [tempdir 'gait2dMEX.' mexext];

    % define the body model (left side), using body segment parameters
    % from Winter, for body mass 75 kg and body height 1.8 m
    % each segment has a local coordinate system that has its origin
    % at the joint where the segment connects to the parent.
    name    = { 'pelvis'    'femur_l'   'tibia_l'   'foot_l'    }';
    parent  = { 'ground'    'pelvis'    'femur_l'   'tibia_l'   }';
    joint   = { 'free'      'hinge'     'hinge'     'hinge'     }';
    mass    = [ 50.850      7.500       3.4875      1.0875      ]';
    inertia = [ 3.1777      0.1522      0.0624      0.0184      ]';
    xcm     = [ 0           0           0           0.0768      ]';    % center of mass position, in local coordinates
    ycm     = [ 0.3155      -0.1910    -0.1917     -0.0351      ]';
    yj      = [ 0           0          -0.4410     -0.4428      ]';    % local y in parent where the joint connects (we require local x zero)

    % add an identical right side
    ileft = find(contains(name,'_l'));
    name    = [name ; strrep(name(ileft),'_l','_r')];
    parent  = [parent ; strrep(parent(ileft),'_l','_r')];
    joint   = [joint ; joint(ileft)];
    mass    = [mass ; mass(ileft)];
    inertia = [inertia ; inertia(ileft)];
    xcm     = [xcm ; xcm(ileft)];
    ycm     = [ycm ; ycm(ileft)];
    yj      = [yj ; yj(ileft)];
    
    % local coordinates of contact points
    xploc = [-0.0600  0.15];    % local x coordinates of heel and toe
    yploc = [-0.0702 -0.0702];  % local y coordinates of heel and toe

    % for each body, determine the DOF indices for its motion
    % relative to the parent, and also the parent segment number
    nSegments = numel(name);
    nDofs = 0;
    for i = 1:nSegments
        % determine the DOF indices iqjoint for segment i
        if strcmp(joint(i), 'free')
            iqjoint{i} = nDofs + (1:3);
            nDofs = nDofs + 3;
            % we could also make indices qall, for all q on which the pose of his body depends
        elseif strcmp(joint(i),'hinge')
            iqjoint{i} = nDofs + 1;
            nDofs = nDofs + 1;
        else
            error('body %s has unrecognized joint type %s', model.name(i), model.joint(i))
        end

        % determine the parent segment number for segment i
        parentnum(i) = -1;
        if strcmp(parent(i), 'ground')
            parentnum(i) = 0;
        else
            for j = 1:nSegments
                if strcmp(parent(i), name(j))
                    parentnum(i) = j;
                    break
                end
            end
        end
        if parentnum(i) == -1
            error('parent %s for % was not found', parent(i), name(i))
        end
    end

    % polygons for stick figure visualization
    polygons = cell(nSegments,1);
    polygons{1} = [0 0 ; 0  0.5184];  % trunk 
    polygons{2} = [0 0 ; 0 -0.4410];  % femur_l
    polygons{3} = [0 0 ; 0 -0.4428];  % tibia_l
    polygons{4} = [ 0       0           ; ...  % foot_l
                   xploc(1) yploc(1)    ; ...
                   xploc(2) yploc(2)    ; ...
                   0        0]; 
    polygons(5:7) = polygons(2:4);  % make a right side copy

    % put some things in a struct that will be returned
    model.nDofs     = 9;
    model.nSegments = 7;
    model.mass = mass;  % segment masses in kg
    model.polygons = polygons;
    model.xploc = xploc;
    model.yploc = yploc;
    model.mex   = funcHandle(mexbinary);

    % Do we need a new MEX function? Yes, if we don't have one yet, or this m file is more recent
    % than the MEX function, or if the MEX source code is more recent than the MEX binary
    thisFolder    = fileparts(mfilename('fullpath'));
    thisFileInfo  = dir([thisFolder filesep mfilename '.m']);
    mexInfo       = dir(mexbinary);
    mexSourceInfo = dir([thisFolder filesep 'gait2dMEX.c']);
    if ~isempty(mexInfo) && ...
            ( datetime(thisFileInfo.date)  < datetime(mexInfo.date) ) && ...
            ( datetime(mexSourceInfo.date) < datetime(mexInfo.date) )
        % if we are here, the mex binary exists, and is not older than
        % this file, and is also not older than the mex source code
        return  % don't continue with the rest of the function
    end

    % define the input variables symbolically
    % (for some reason, the Jacobians don't simplify correctly if I do: q = sym('q',[nDofs 1],'real') )
    syms   q [nDofs 1] real
    syms  qd [nDofs 1] real
    syms qdd [nDofs 1] real
    syms   v real

    % generate equations for forward kinematics: x,y,a as a function of q
    fprintf('   Generating expressions for forward kinematics and inverse dynamics...\n')
    for i = 1:nSegments
        % get pose of parent of body i
        if parentnum(i) == 0  % ground
            xp = 0;
            yp = 0;
            ap = 0;
        else
            xp = x(parentnum(i));
            yp = y(parentnum(i));
            ap = a(parentnum(i));
        end

        % generate expressions for pose of body i
        if strcmp(joint(i), 'free')
            x(i,1) = xp + q(iqjoint{i}(1));
            y(i,1) = yp + q(iqjoint{i}(2));
            a(i,1) = ap + q(iqjoint{i}(3));
        elseif strcmp(joint(i), 'hinge')
            x(i,1) = xp - yj(i)*sin(ap);
            y(i,1) = yp + yj(i)*cos(ap);
            a(i,1) = ap + q(iqjoint{i});
        end
    end

    % generate Jacobian matrices for the forward kinematics expressions x,y,a
    x_q = jacobian(x,q);
    y_q = jacobian(y,q);
    a_q = jacobian(a,q);

    % generate accelerations xcmdd, ycmdd, add for all body segments
    % global coordinates of the center of mass (for comment only, we don't need them)
    % xcmg = x + xcm.*cos(a) - ycm.*sin(a);   
    % ycmg = y + xcm.*sin(a) + ycm.*cos(a);
    % jacobians of xcmg and ycmg with respect to q
    xcm_q = x_q + a_q .* (-xcm.*sin(a) - ycm.*cos(a));
    ycm_q = y_q + a_q .* ( xcm.*cos(a) - ycm.*sin(a));
    % cm velocities and segment angular velocities:
    xcmd = xcm_q * qd;
    ycmd = ycm_q * qd;
    ad = a_q * qd;
    % and finally the cm acceleration and segment angular accelerations:
    xcmdd = xcm_q * qdd + jacobian(xcmd,q) * qd;
    ycmdd = ycm_q * qdd + jacobian(ycmd,q) * qd;
    add   = a_q   * qdd + jacobian(ad,q) * qd;

    % generate forward kinematics output (FK) and Jacobian dFK/dq
    % FK is a column vector with x,y,a for all segments
    FK = reshape([x y a]', 3*nSegments, 1);
    % we could now simply do: FK_q = jacobian(FK,q);
    % but I would rather reuse expressions we already have stored
    syms FK_q [3*nSegments nDofs] real
    for i = 1:nSegments
        FK_q(3*i-2, :) = x_q(i,:);
        FK_q(3*i-1, :) = y_q(i,:);
        FK_q(3*i  , :) = a_q(i,:);
    end

    % divide mass and inertia by 1000, so we get the inertial and
    % gravitational generalized forces in kN and kNm
    mass = mass/1000;
    inertia = inertia/1000;

    % calculate inertia part of Q using Kane's equation  
    Q = xcm_q' * (mass .* xcmdd) + ...  % in components: Q(j) = sum(m(i) * xcmdd(i) * d(xcm(i)/d(q(j)) )
        ycm_q' * (mass .* ycmdd) + ...
        a_q'   * (inertia .* add);

    % parameters for model of external forces applied to the heel and toe
    % of each foot
    xploc = [-0.0600  0.15];    % local x coordinates of heel and toe
    yploc = [-0.0702 -0.0702];  % local y coordinates of heel and toe
    kc = 1e4;       % ground contact stiffness, kN/m^2
    kclin = 0.001;  % linear stiffness, kN/m
    cc = 0.85;      % ground contact damping, s/m
    mu = 1.0;       % friction coefficient
    vs = 0.1;       % velocity constant (m/s), for |ve| = vs -> |fx| = 0.4621*c*fy
    contact_segments = [4 7];

    % initialize a 6x1 matrix for GRF output
    syms GRF [6 1] real 
    GRF(:) = 0;

    % generate the friction and contact forces, and subtract their
    % generalized force contributions from Q
    for iseg = 1:numel(contact_segments)
        seg = contact_segments(iseg);
        for ip = 1:2
            % calculate global coordinates and velocities of contact point ip on seg
            xp = x(seg) + xploc(ip) * cos(a(seg)) - yploc(ip) * sin(a(seg));
            yp = y(seg) + xploc(ip) * sin(a(seg)) + yploc(ip) * cos(a(seg));
            xp_q = x_q(seg,:) + a_q(seg,:) * (-xploc(ip)*sin(a(seg)) - yploc(ip)*cos(a(seg)));
            yp_q = y_q(seg,:) + a_q(seg,:) * ( xploc(ip)*cos(a(seg)) - yploc(ip)*sin(a(seg)));
            xpd = xp_q * qd;
            ypd = yp_q * qd;

            % contact force
            penetration = (abs(yp) - yp) / 2;
            Fy = (kc * penetration^2 - kclin * yp) * (1 - cc * ypd);

            % friction force
            % v is speed of the treadmill belt, which moves in the negative x direction
            Fx = -mu * Fy * ((2 / (1 + exp( (-v - xpd) / vs))) - 1);
            
            % alternative (algebraic) sigmoid function
            % vs = 0.01;
            % Fx = -mu * Fy * (xpd + v) / sqrt((xpd+v)^2 + vs^2);

            % subtract the generalized forces, generated by F, from Q
            Q = Q - xp_q' * Fx - yp_q' * Fy;

            % create GRF output
            GRF((iseg-1)*3 + 1) = GRF((iseg-1)*3 + 1) + Fx;
            GRF((iseg-1)*3 + 2) = GRF((iseg-1)*3 + 2) + Fy;
            GRF((iseg-1)*3 + 3) = GRF((iseg-1)*3 + 3) + xp*Fy - yp*Fx;

        end
    end

    % subtract the contributions from gravity, acting at the CM of each segment
    g = 9.81;
    Q = Q + ycm_q' * mass * g;   % we use a positive value for the constant g, so we are subtracting the effect of a -mg vertical force

    % calculate the jacobians of Q
    fprintf('   Generating dynamics Jacobians...\n')
    Q_q   = jacobian(Q,q);
    Q_qd  = jacobian(Q,qd);
    Q_qdd = jacobian(Q,qdd);

    % for generating C code, combine all variables into a column vector v
    Q_q   = reshape(Q_q,  nDofs^2,1);
    Q_qd  = reshape(Q_qd, nDofs^2,1);
    Q_qdd = reshape(Q_qdd,nDofs^2,1);
    FK_q  = reshape(FK_q, 3*nSegments*nDofs,1);
    if options.contactBuiltin
        var      = [Q;      Q_q ;    Q_qd ;  Q_qdd;   FK;          GRF];
        varnames = {'Q',   'Q_q',   'Q_qd', 'Q_qdd', 'FK',        'GRF'};
        varsizes = [nDofs, nDofs^2, nDofs^2, nDofs^2, 3*nSegments, 6];
    else
        var      = [Q;     Q_q ;    Q_qd ;   Q_qdd;   Q_G;     FK;          FK_q];
        varnames = {'Q',  'Q_q',   'Q_qd',  'Q_qdd'; 'Q_G';   'FK';        'FK_q'};
        varsizes = [nDofs, nDofs^2, nDofs^2, nDofs^2, 6*nDofs, 3*nSegments, 3*nSegments*nDofs];
    end

    % for each element of var, generate the name and element of the matrix where it should be stored
    % and the array index for the C code
    nv = 0;
    for ivar = 1:numel(varnames)
        for i = 1:varsizes(ivar)
            nv = nv+1;
            names{nv} = varnames{ivar};
            index(nv) = i-1;
        end
    end

    fprintf('   Generating C code...')
    ccode(var, 'file', 'tmp0.c');
    fprintf('(%d bytes generated for Q, GRF, and Jacobians)\n', dir('tmp0.c').bytes)

    % first, create some C code to copy the matlab inputs into C variables
    % q1 = q[0] etc.
    fid2 = fopen("tmp1.c","w");
    for i = 1:nDofs
        fprintf(fid2,'double q%d = q[%d];\n', i, i-1);
        fprintf(fid2,'double qd%d = qd[%d];\n', i, i-1);
        fprintf(fid2,'double qdd%d = qdd[%d];\n', i, i-1);
    end
    fprintf(fid2,'double v = *vptr;\n');

    % keep track of the last element of var that was generated in the C code
    lastGenerated = -1;

    % alter the C code, generating tmp1.c which can be inserted into gait2dMEX.c
    fid1 = fopen("tmp0.c");
    while ~feof(fid1) 
        t = fgetl(fid1);  % get one line
        t = strip(t);     % strip off leading and trailing spaces

        % process the line, depending on whether it starts with A0[number] or not
        k = sscanf(t,'A0[%d');
        if ~isempty(k)
            % TODO: at the right place, insert "if (nlhs < ...) return" to skip Jacobian
            % calculation code when Jacobian is not needed

            % for any preceding elements of v that were zero (and not defined in the C code)
            % we need to generate extra lines: name[index] = 0.0
            for i = (lastGenerated+1):(k-1)
                fprintf(fid2,'%s[%d] = 0.0;\n', names{i+1}, index(i+1));
            end
            % now replace A0[k][0] by name[index])
            p = findstr(t,'] ');
            fprintf(fid2, '%s[%d] %s\n', names{k+1}, index(k+1), t((p+1):end));
            lastGenerated = k;
        else
            % if not starting with A0[, this is a line that creates and 
            % computes a temporary intermediate variable.
            % put "double" in front of the line to declare the variable
            fprintf(fid2, 'double %s\n', t);
        end
    end
    fclose(fid2);
    fclose(fid1);

    % compile gait2dMEX.c
    % (gait2dMEX.c has a line "#include tmp1.c" to insert the generated C code)
    fprintf('   Compiling MEX function...')
    mex('gait2dMEX.c','-silent','-output',mexbinary);
    fprintf('   (done). gait2dMEX.%s is ready to use.\n', mexext)

    % clean up (comment out if you want to see the generated C code)
    delete tmp0.c
    delete tmp1.c
    
end
%====================================================================
function [fh] = funcHandle(filePath)
    % make a function handle for a function that is in a different folder
    % that is not in the matlab path
    tmp = pwd;  % save the current directory
    [folder,name] = fileparts(filePath);
    cd(folder);
    fh = str2func(name);
    cd(tmp);    % go back to where we were
end