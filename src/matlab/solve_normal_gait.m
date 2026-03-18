function solve_normal_gait()

    rng(0) % for reproducible results
    close all

    % load the model (gait2d or gait2dc)
    model = gait2d();
    % initial settings
    problem.N = 50;
    % problem.N = 3;  % N=3 will perform derivative checking
    problem.Wtor = 100;  % 100
    problem.Wang = 100;  % 100
    problem.Wreg = 1e-6;    % 1e-5
    problem.duration = 1.1396/2;  % using Winter normal data
    problem.conScale = 10.0;

    % variables in each node are the states and controls
    problem.nvarpernode = model.nx + model.nu;

    % define indices to perform the symmetry operation
    problem.sym = [model.xsym ; model.nx + model.usym];

    % store model in problem
    problem.model = model;
   
    % solve a series of problems
    initial_guess = 'standing';
    % initial_guess = 'random';
    nSpeeds = 10;
    normal_speed = 1.325;  % speed at which Winter's data was collected
    speeds = linspace(0.1, normal_speed, nSpeeds);
    for i = 1:numel(speeds)
        problem.speed = speeds(i);
        result = solve(problem, initial_guess);
        gaitReport(result);
        initial_guess = result;
    end

    % movie
    gaitMovie(result)

end
%===================================================
function result = solve(problem, initial_guess)

    N = problem.N;
    model = problem.model;

    % make a struct with tracking data
    problem.data = makeTrackingData(model,N);

    % set number of unknowns and number of constraints
    nX = N * problem.nvarpernode;
    problem.nX = nX;
    nC = (N-1) * model.nf + problem.nvarpernode;  % dynamics and periodicity constraints
    problem.nC = nC;

    % make a list of X indices for quick objective function calculation
    problem.iangles = [];
    problem.itorques = [];
    problem.regi1 = [];
    problem.regi2 = [];
    for i = 1:(N-1)  % do not include the final time point (it's equal to the first)
        problem.iangles  = [problem.iangles ; (i-1)*problem.nvarpernode + (4:9)'];
        problem.itorques = [problem.itorques ; (i-1)*problem.nvarpernode + model.nx + (1:model.nu)'];
        problem.regi1 = [problem.regi1 ; (i-1)*problem.nvarpernode + (1:problem.nvarpernode)'];
        problem.regi2 = [problem.regi2 ;     i*problem.nvarpernode + (1:problem.nvarpernode)'];
    end

    % set appropriate bounds for x and u
    lbq = [-1 0.5 pi/180*[-45 -90 -120 -40 -90 -120 -40] ];
    ubq = [ 1 1.5 pi/180*[ 45  90  0    40  90  0    40] ];
    lbqdot = [-2 -2 -pi/180*[1000 1000 1000 1000 1000 1000 1000] ];
    ubqdot = [ 2  2  pi/180*[1000 1000 1000 1000 1000 1000 1000] ];
    lbx = [lbq lbqdot]';
    ubx = [ubq ubqdot]';

    % set bounds for controls
    lbu = -0.4;
    ubu = 0.4;
    lb = repmat([lbx ; lbu+zeros(model.nu,1)], N, 1);
    ub = repmat([ubx ; ubu+zeros(model.nu,1)], N, 1);

    % constrain horizontal position at node 1, otherwise there is no unique solution
    lb(1) = 0;
    ub(1) = 0; 

    % ipopt settings
    funcs.objective   = @objGait;
    funcs.gradient    = @gradGait;
    funcs.constraints = @conGait;
    funcs.jacobian    = @conjacGait;  
    options.lb = lb;
    options.ub = ub;
    options.cl = zeros(nC,1);
    options.cu = zeros(nC,1);
    options.ipopt.print_level = 0;
	options.ipopt.hessian_approximation = 'limited-memory'; 
    options.ipopt.tol = 1e-3;  % Desired convergence tolerance.
    options.ipopt.constr_viol_tol = 1e-4; % Absolute tolerance on the constraint violation. 

    % initial guess from standing
    if (nargin < 2) || strcmp(initial_guess,'standing')
        s = load('standing-straight');
        X0 = repmat([s.x ; s.u], N, 1);
    else
        X0 = initial_guess.X;
    end

    % determine sparsity structure of the constraints Jacobian
    problem.Jpattern = spalloc(nC, nX, 10);
    problem.Jpattern = conjacGait(X0 + 0.001*rand(size(X0)), problem);
    funcs.jacobianstructure = @(a1,a2) problem.Jpattern; 
 
    % if N=3, we check the objective gradient and constraint Jacobian
    if (N == 3)
        X = X0;
        hh = 1e-8;
        f = objGait(X,problem);
        g = gradGait(X,problem);
        gnum = zeros(size(g));
        for i = 1:nX
            tmp = X(i);
            X(i) = X(i) + hh;
            fhh = objGait(X,problem);
            gnum(i) = (fhh-f)/hh;
            X(i) = tmp;
        end
        [e,index] = max(abs(g-gnum));
        fprintf("largest error in gradient is %e at element %d (%.14f vs %.14f)\n", ...
            e,index,g(index),gnum(index));

        c = conGait(X,problem);
        J = conjacGait(X,problem);
        Jnum = problem.Jpattern;
        for i = 1:nX
            tmp = X(i);
            X(i) = X(i) + hh;
            chh = conGait(X,problem);
            Jnum(:,i) = (chh-c)/hh;
            X(i) = tmp;
        end
        [~,col] = max(max(abs(J-Jnum)));
        [e,row] = max(max(abs(J'-Jnum')));
        maxerr = full(e);
        fprintf("largest error in Jacobian is %e at %d,%d (%.14f vs %.14f)\n", ...
            maxerr,row,col,full(J(row,col)),full(Jnum(row,col)));
        keyboard
    end

    % solve
    fprintf('solving %.3f m/s...', problem.speed)
    options.auxdata = problem;
    [X, info] = ipopt_auxdata(X0, funcs, options);
    fprintf('done. iter=%4d, obj=%.3f, cpu=%6.3f\n', info.iter, info.objective, info.cpu)
    if (info.status < 0)
        fprintf('Optimal gait was not solved.\n')
        keyboard
    end

    % make a result structure and save on file
    result.problem = problem;
    result.X = X;
    result.info = info;  % info from IPOPT
end
% =============================================================
function f = objGait(X, problem)

    % torque cost, angle tracking cost and regularization cost
    f_tor = problem.Wtor * mean(X(problem.itorques).^2);
    f_ang = problem.Wang * mean((X(problem.iangles)-problem.data.angles).^2);

    % regularization cost: mean of squared first derivatives
    regi1 = problem.regi1;
    regi2 = problem.regi2;
    h = problem.duration / (problem.N-1);
    f_reg = problem.Wreg * mean((X(regi2)-X(regi1)).^2) / h^2;

    % total cost
    f = f_tor + f_ang + f_reg;
    pause(0.001)
    fprintf('  obj: %.4f = %.4f(torque) + %.4f(ang) + %.4f(reg)\n', f,f_tor,f_ang,f_reg)

end
%===================================
function g = gradGait(X,problem)
    g = zeros(size(X));

    % torque cost
    % f_tor = problem.Wtor * mean(X(problem.itorques).^2);
    g(problem.itorques) = problem.Wtor * 2 * X(problem.itorques) / numel(problem.itorques);    

    % angle cost
    % f_ang = problem.Wang * mean((X(problem.iangles)-problem.data.angles).^2);
    g(problem.iangles) = ...
        problem.Wang * 2 * (X(problem.iangles)-problem.data.angles) / numel(problem.iangles); 

    % f_reg = problem.Wreg * mean((X(ix2)-X(ix1)).^2) / h^2;
    regi1 = problem.regi1;
    regi2 = problem.regi2;
    h = problem.duration / (problem.N-1);
    g(regi1) = g(regi1) - 2*problem.Wreg*(X(regi2)-X(regi1)) / numel(regi1) / h^2;
    g(regi2) = g(regi2) + 2*problem.Wreg*(X(regi2)-X(regi1)) / numel(regi1) / h^2;

end
%===================================
function c = conGait(X,problem)

    % extract some things from problem
    N = problem.N;
    model = problem.model;
    speed = problem.speed;
    conScale = problem.conScale;
    h = problem.duration / (N-1);

    % initialize
    c = zeros(problem.nC,1);

    % states and controls in node 1
    ix = 1:model.nx;
    x1 = X(ix);
    iu = model.nx + (1:model.nu);

    % calculate the dynamics constraints for the N-1 time steps
    % each time step advances states and controls from (x1,u1) to (x2,u2)
    ic = 1:model.nf;
    for istep = 1:(N-1)
        % extract x,u at end of time step
        ix = ix + problem.nvarpernode;
        x2 = X(ix);
        iu = iu + problem.nvarpernode;
        u2 = X(iu);
    
        % evaluate the model dynamics using backward Euler and store in c
        c(ic) = conScale * model.dynamics(model,x2,(x2-x1)/h,u2,speed);

        % advance to the next time step
        ic = ic + model.nf;
        x1 = x2;
    end

    % add the periodicity constraints for a half gait cycle
    % require state at node 1 to be equal to node N (after applying symmetry operation)
    ic = (N-1) * model.nf + (1:problem.nvarpernode)';
    iN = (N-1) * problem.nvarpernode + (1:problem.nvarpernode)';
    c(ic) = X(iN) - X(problem.sym);

end
%===================================
function J = conjacGait(X,problem)  

    % extract some things from problem
    N = problem.N;
    model = problem.model;
    speed = problem.speed;
    conScale = problem.conScale;
    h = problem.duration / (N-1);

    % initialize the sparse matrix
    J = problem.Jpattern;

    % indices to states in node 1, and controller parameters
    ix1 = 1:model.nx;
    x1 = X(ix1);
    iu = model.nx + (1:model.nu);

    % indices to constraints in first time step
    ic = 1:model.nf;

    % calculate the dynamics constraints for the N-1 time steps
    % each time step advances states and controls from (x1,u1) to (x2,u2)
    for istep = 1:(N-1)
        % extract x,u at end of time step
        ix2 = ix1 + problem.nvarpernode;
        x2 = X(ix2);
        iu = iu + problem.nvarpernode;
        u2 = X(iu);
    
        % evaluate the model dynamics jacobian using backward Euler and store in J
        [~,df_dx,df_dxdot,df_du] = model.dynamics(model,x2,(x2-x1)/h,u2,speed);
        J(ic,ix1) = -conScale * df_dxdot/h;
        J(ic,ix2) = conScale * (df_dx + df_dxdot/h);
        J(ic,iu) = conScale * df_du;

        % advance to the next time step
        ic = ic + model.nf;
        ix1 = ix2;
        x1 = x2;
    end

    % add the periodicity constraints for a half gait cycle
    ic = (N-1) * model.nf + (1:problem.nvarpernode)';
    iN = (N-1) * problem.nvarpernode + (1:problem.nvarpernode)';
    % c(ic) = X(iN) - X(problem.sym);
    J(ic,iN)        =  speye(problem.nvarpernode);
    J(ic,problem.sym) = -speye(problem.nvarpernode);

end
%======================================================================
function gaitMovie(result)

    % extract information from solution
    problem = result.problem;
    model = problem.model;
    N = result.problem.N;
    h = result.problem.duration / (N-1);
    X = result.X;

    % extract the state trajectory
    xu = X(1:(N*problem.nvarpernode));
    xu = reshape(xu, problem.nvarpernode, N);
    x = xu(1:model.nx, :);

    % create second half of the gait cycle
    % and add it to the first half, then copy the full gait cycle a few
    % more times
    x2 = x(model.xsym, 2:end);
    x = [x x2];

    % and double it two more times to get 4 full gait cycles
    x = [x x(:,2:end)];
    x = [x x(:,2:end)];

    % initialize the video file
    figure(2);
    clf
    set(gcf,'Position',[360 220 460 300]);
    xFloor = -0.8 + 1.6*rand(2000,1);
    yFloor = -0.1 + 0.1*rand(2000,1);
    video = VideoWriter('normal_gait', 'MPEG-4');
    interval = round(N/25);
    lastframe = size(x,2);  % show all frames
    avSpeed = (x(1,end) - x(1,1)) / size(x,2);  % how much the model moves forward in each time step
    
    % make a movie with the full model
    open(video);
    for i = 1:interval:lastframe
        timeValue = (i-1)*h;
        model.visualize(model,x(:,i));

        % show COP and resultant force vector in each foot
        g = model.GRF(model,x(:,i),result.problem.speed);
        COPl = g(3)/g(2);  % COPx = Mz / Fy;
        COPr = g(6)/g(5);
        quiver(COPl,0,g(1)/2,g(2)/2,'k');  % scale will be 1 m for 2 kN
        quiver(COPr,0,g(4)/2,g(5)/2,'k');

        % TODO: make floor move to the left at belt speed
        scatter(xFloor,yFloor,1,'k.');

        title([' t = ' num2str(timeValue,'%.2f')], 'FontSize', 14);
        set(gca,'XTick',[]);  % this prevents the axes from jumping around
        axis([-0.8 0.8 -0.2 1.5])

        frame = getframe(gcf);
        writeVideo(video,frame);
    end

    % close the file
    close(video);

end
%======================================================================
function gaitReport(result)

    % if result is a filename, get the result (struct) by loading the file
    if isstr(result)
        load(result);
    end

    % from result struct, extract model, and trajectory
    problem = result.problem;
    model = problem.model;
    X = result.X;
    N = result.problem.N;

    % extract states and controls
    xu = X(1:(N*problem.nvarpernode));
    xu = reshape(xu,problem.nvarpernode,N);

    % create second half of the gait cycle and add it to the first half
    xu = [ xu xu(problem.sym, 2:end)];
    x = xu(1:model.nx,:);
    u = xu(model.nx+(1:model.nu),:);

    % do that also for the tracking data
    a = reshape(problem.data.angles,6,N-1)';
    a = [a ; a(1,[4 5 6 1 2 3])];  % add values at 100% of gait cycle, mirror image of 0%
    angdata = 180/pi * [a(:,1:3) ; a(2:end,4:6)];
    g = problem.data.grf;
    g = [g ; g(1,[3 4 1 2])];
    grfdata = 1000 * [g(:,1:2) ; g(2:end,3:4)];

    % make the plots
    figure(1);
    clf
    set(gcf,'Position',[898 158 379 483]);
    n = 2*N-1;
    tperc = 100*(0:(n-1))/(n-1);

    subplot(3,1,1)
    angles = 180/pi*x([4 5 6 3],:)'; % hip, knee, ankle, pelvis
    angles(:,2)  = -angles(:,2);  % invert knee angle so we see flexion angle
    angdata(:,2) = -angdata(:,2);  % invert knee angle so we see flexion angle
    p1 = plot(tperc,angles);
    hold on
    p2 = plot(tperc,angdata,'--');
    [p2.Color] = deal(p1(1:3).Color);
    xlabel('time (% gait cycle)');
    ylabel('flexion angle (deg)');
    legend('hip','knee','ankle','pelvis tilt');

    subplot(3,1,2)
    G = zeros(6,n);
    for i = 1:n
        G(:,i) = model.GRF(model,x(:,i),result.problem.speed);
    end
    p1 = plot(tperc, 1000*G([1 2],:)');
    hold on
    p2 = plot(tperc, grfdata,'--');
    [p2.Color] = deal(p1.Color);
    xlabel('time (% gait cycle)');
    ylabel('force (N)');
    legend('Fx','Fy');
    
    subplot(3,1,3)
    M = 1000*u(1:3,:)';  % torques in Nm
    M(:,[1 3]) = -M(:,[1 3]);  % invert hip and ankle torques, so we see extensor/plantarflexor torque
    plot(tperc,M);
    ylabel('extensor torque (Nm)');
    legend('hip','knee','ankle');
    xlabel('time (% gait cycle)');

    shg;

end
%=====================================================================
function [data] = makeTrackingData(model,N)

    % gait data from Winter, normal cadence
    load('C:\Users\Ton\Documents\Literature\Books\Biomechanics\Winter 1991 data\Winter_normal.mat');

    % make a half gait cycle, left and right
    Ndata = size(gait.data,1);
    if Ndata ~= 51
        error('makeTrackingData: code requires 51 samples for the gait cycle.')
    end
    i50 = 26;  % sample number at 50% of gait cycle
    dat = [gait.data(1:i50,:) gait.data(i50:end,:)];
    ang = pi/180 * dat(:,[1:3 9:11]);
    ang(:,[2 5]) = -ang(:,[2 5]);         % invert the knee angle so it is consistent with q(5) in the model

    % extract left and right GRF, convert from N/kg to kN
    grf = dat(:,[4:5 12:13]);
    grf = 0.001 * grf * sum(model.mass);

    % interpolate to N sanmples
    t1 = (0:(i50-1))/(i50-1);   % original time values from 0 to 1
    t2 = (0:(N-1))/(N-1);       % resample times from 0 to 1
    ang = interp1(t1,ang,t2)';  % 6 rows, N columns
    ang = ang(:,1:end-1);       % remove time point N, it's 100% and should not be counted twice
    grf = interp1(t1,grf,t2)';  % 6 rows, N columns
    grf = grf(:,1:end-1);       % remove time point N, it's 100% and should not be counted twice

    % reshape to a column matrix in the same order as angles appear in free variables X
    % and store in data struct
    data.angles = reshape(ang, (N-1)*6, 1);
    data.grf = grf';

    % also store the average speed and gait cycle duration 
    data.speed = gait.speed(1);
    data.dur   = gait.dur(1);

end
