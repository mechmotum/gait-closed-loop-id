function model = gait2d()

    % make the MEX function if needed
    model = makeGait2dMEX;

    % add size constants to the model struct
    model.nx = 18;
    model.nf = 18;
    model.nu = 6;

    % symmetry indices to switch left and right
    qsym = [1 2 3 7 8 9 4 5 6]';
    model.xsym = [qsym ; model.nDofs+qsym];
    model.usym = [4 5 6 1 2 3]';

    % add function handles
    model.visualize = @visualize;
    model.dynamics  = @dynamics;
    model.GRF       = @GRF;

end
%==========================================================
function G = GRF(model,x,v)
    q = x(1:model.nDofs);
    qd = x(model.nDofs + (1:model.nDofs));
    qdd = zeros(model.nDofs,1);  % accelerations dont matter for GRF calculation
    [~,~,~,~,~,G] = model.mex(q,qd,qdd,v);
end
%==========================================================
function visualize(model,x)
% show the model

    clf
    hold on

    % extract the generalized coordinates
    q = x(1:model.nDofs);

    % use forward kinematics to draw all segments
    [~,~,~,~,FK] = model.mex(q);
    FK = reshape(FK,3,model.nSegments);
    for i = 1:model.nSegments
        ps = FK(1:2,i);  % x and y of the segment
        theta = FK(3,i);
        R = [cos(theta) -sin(theta); sin(theta) cos(theta)];
        % transform the polygon points to global coordinates
        pp = ps + R * model.polygons{i}';

        % draw the polygon, with trunk and left leg blue, right leg red
        if (i<5)
            plot(pp(1,:),pp(2,:),'b');
        else
            plot(pp(1,:),pp(2,:),'r');
        end
    end

    % finalize the figure
    axis('equal');
    grid on; box on;

end
%==========================================================
function [f,df_dx,df_dxd,df_du] = dynamics(model,x,xd,u,v)

% gait2d first order implicit dynamics f(x,xdot,u,v) = 0

    % model size
    nx = 18;
    nu = 6;
    nf = 18;

    % if no speed is provided, assume zero
    if (nargin < 5)
        v = 0.0;
    end

    % initialize the outputs
    f = zeros(nf,1);
    if nargout > 1
        % TODO: we could have preallocated these matrices with their
        % sparsity pattern, for better speed
        df_dx  = zeros(nf,nx); 
        df_dxd = zeros(nf,nx); 
        df_du  = zeros(nf,nu);
    end

    % run the MEX function to calculate the generalized forces Q
    % that are required for motion q,qd,qdd, while belt speed is v
    nq = 9;
    iq = 1:nq;
    iqd = nq + (1:nq);
    q   = x(iq);
    qd  = x(iqd);
    qdd = xd(iqd);
    if nargout == 1
        Q = model.mex(q,qd,qdd,v);
    else
        [Q,dQ_dq,dQ_dqd,dQ_dqdd] = model.mex(q,qd,qdd,v);
    end

    % first set of equations (rows): qd - dq/dt = 0
    rows = (1:nq);
    f(rows)  = qd - xd(iq);  % qd - dq/dt 
    if (nargout > 1)
        df_dx(rows, iqd) = eye(nq);   % df/dqd = 1
        df_dxd(rows, iq) = -eye(nq);  % df/dqd = -1
    end

    % next set of rows: Qrequired(q,qd,qdd) - Qapplied = 0;
    rows = nq + (1:nq);
    f(rows) = Q - [0;0;0;u];  % first 3 DOFs are not actuated
    if (nargout > 1)
        df_dx(rows,iq)   = dQ_dq;
        df_dx(rows,iqd)  = dQ_dqd;
        df_dxd(rows,iqd) = dQ_dqdd;  
        df_du(rows(4:end),:) = -eye(nu);
    end

    % output sparse Jacobians
    % TODO: the "sparse" operations would not be necessary if we
    % had already initialized the matrices with the sparsity pattern
    if (nargout > 1)
        df_dx  = sparse(df_dx);
        df_dxd = sparse(df_dxd);
        df_du  = sparse(df_du);
    end

end



