function gait2d_test
% test the gait2d model

    clear all
    close all
    rng(0)

    % make the model
    model = gait2d();

    testVisualization(model);
    testGRF(model);
    testDynamics(model);
    testDynamicsDerivatives(model);
    testStanding(model, 'straight');
    testStanding(model, 'flexed');
    testSimulation(model);

end
%======================================================================
function testGRF(model)

    fprintf('\nTesting gait2d GRF model:\n')

    x = zeros(model.nx,1);
    x(7) = pi/2;  % lift the right leg, so only left will produce GRF

    % plot vertical force as a function of penetration
    dx2 = -0.015:0.0001:0.015;
    for i = 1:numel(dx2)
        x(2) = 0.954 - dx2(i); % 0.954 puts the contact points exactly at y=0
        G = model.GRF(model,x,0);
        Fy(i) = G(2);  % Fy on left leg
    end
    figure
    subplot(1,2,1);
    plot(dx2,1000*Fy);
    xlabel('penetration');
    ylabel('Fy (N)')

    % plot horizontal force as a function of belt speed
    x(2) = 0.954 - 0.01;  % 1 cm penetration
    v = -1:0.001:1;
    for i = 1:numel(v)
        G = model.GRF(model,x,v(i));
        Fx(i) = G(1);  % Fx on left leg
    end
    subplot(1,2,2);
    plot(v,1000*Fx);
    xlabel('belt speed (m/s)');
    ylabel('Fx (N)');

end
%======================================================================
function testVisualization(model)

    fprintf('\nTesting gait2d visualization:\n')

    % create a slightly random state
    rng(0);
    x = 0.1 * rand(model.nx,1);  
    x(2) = 1.0;  % raise the model above ground
    
    % display the model 
    figure
    model.visualize(model,x);  

end
%=====================================================================
function testDynamics(model)
    fprintf('\nTesting gait2d dynamics...\n')
    rng(0);

    % generate a freefall state
    x = zeros(model.nx,1);
    x(2) = 0.954;  % places the contact points exactly at y=0, so Fy=0
    xd = zeros(model.nx,1);
    xd(model.nDofs + 2) = -9.81;  % vertical acceleration
    u = zeros(model.nu,1);
    
    % calculate dynamics residuals f, they should be close to zero
    speed = 0.0;
    f = model.dynamics(model,x,xd,u,speed);
    f_expected = zeros(size(f));
    fprintf('rms dynamics error: %e\n', rms(f-f_expected))

    % measure how fast the dynamics function is for this model
    x = rand(model.nx,1);
    xd = rand(model.nx,1);
    u = rand(model.nu,1);
    nEval = 0;
    tic
    while toc < 2.0
        f = model.dynamics(model,x,xd,u,speed);
        nEval = nEval + 1;
    end
    fprintf('gait2d: %.2f function calls per second\n', nEval / toc);

end
%=====================================================================
function testDynamicsDerivatives(model)

    fprintf('\nTesting gait2d dynamics derivatives...\n')
    rng(0);

    % first measure how fast the dynamics function is for this model,
    % when it needs to calculate derivatives
    x = rand(model.nx,1);
    xd = rand(model.nx,1);
    u = rand(model.nu,1);
    speed = rand();
    nEval = 0;
    tic
    while toc < 2.0
        [f,df_dx,df_dxd,df_du] = model.dynamics(model,x,xd,u,speed);
        nEval = nEval + 1;
    end
    fprintf('%25s: function calls per second = %.2f\n', 'dynamics+jacobians', nEval / toc);

    % generate random states, state velocities, and controls
    x = zeros(model.nx,1);
    x(2) = 1.0;  % this should generate about 1 BW of vertical force
    xd = 0.1*randn(model.nx,1);
    u = randn(model.nu,1);

    % get the dynamics residuals, with jacobians, and initialize the numerical jacobians
    [f,df_dx,df_dxd,df_du] = model.dynamics(model,x,xd,u,speed);
    df_dx_num  = zeros(size(df_dx));
    df_dxd_num = zeros(size(df_dxd));
    df_du_num  = zeros(size(df_du)); 

    % show the sparsity patterns
    figure
    subplot(2,2,1); spy(df_dx);  title('gait2d df/dx')
    subplot(2,2,2); spy(df_dxd); title('gait2d df/dxdot')
    subplot(2,2,3); spy(df_du);  title('gait2d df/du')

    % calculate the numerical approximations with finite differences
    hh = 1e-7;
    for i = 1:model.nx
        tmp = x(i);
        x(i) = x(i) + hh;
        fhh = model.dynamics(model,x,xd,u,speed);
        df_dx_num(:,i) = (fhh - f)/hh;
        x(i) = tmp;

        tmp = xd(i);
        xd(i) = xd(i) + hh;
        fhh = model.dynamics(model,x,xd,u,speed);
        df_dxd_num(:,i) = (fhh - f)/hh;
        xd(i) = tmp;
    end
    for i = 1:model.nu
        tmp = u(i);
        u(i) = u(i) + hh;
        fhh = model.dynamics(model,x,xd,u,speed);
        df_du_num(:,i) = (fhh - f)/hh;
        u(i) = tmp;
    end
    matcompare(df_dx, df_dx_num,   'df/dx'); 
    matcompare(df_dxd,df_dxd_num, 'df/dxd');
    matcompare(df_du, df_du_num,   'df/du');

end
%=====================================================================
function matcompare(a,b,name,tol)

    if (nargin < 4)
        tol = 1e-6;
    end

    % compares two matrices
    % prints RMS diff and element that has greatest relative difference
    err = a-b;
    [maxerr,irow] = max(abs(err));
    [maxerr,icol] = max(maxerr);
    irow = irow(icol);
    maxab = max(abs(a),abs(b));
    relerr = maxerr / maxab(irow,icol);  % relative error, at the element where absolute error is largest
    err = min(maxerr,relerr);

    fprintf('%25s:', name)
    if (err > tol)

        fprintf('\n   Largest error: %12.5e at %d %d (%22.15e should be %22.15e)\n', full(maxerr), irow, icol, full(a(irow,icol)), full(b(irow,icol)));
        fprintf('   Relative error at this element: %12.5e\n', full(relerr));
        keyboard
        error('difference is above tolerance (%e)', tol)
    else
        fprintf(' within tolerance.\n')
    end
    
end  
%======================================================================
function testStanding(model, pose)
% find an equilibrium state with minimal control effort

    fprintf('\nFinding optimal standing...\n')
    global problem
    problem.model = model;

    % scale everything to N
    problem.objScale = 1000;  % to make objective equivalent to the python version (which had Nm units)
    problem.conScale = [ones(9,1); 1000*ones(9,1)];
    problem.conScale = sparse(diag(problem.conScale));
    
    % set up indices for unknowns X
    problem.h = 1.0;  % time step (doesn't matter)
    nX = 2*(model.nx + model.nu);
    problem.iu = [model.nx + (1:model.nu)'; 2*model.nx + model.nu + (1:model.nu)'];

    % determine sparsity structure of the constraints Jacobian
    X = rand(nX,1);
    problem.Jpattern = spalloc(model.nf, model.nx + model.nu,1);
    problem.Jpattern = conjacStanding(X);

    % check Jacobian
    c = conStanding(X);
    J = conjacStanding(X);
    Jnum = zeros(size(J));
    hh = 1e-7;
    for i = 1:nX
        tmp = X(i);
        X(i) = X(i) + hh;
        chh = conStanding(X);
        Jnum(:,i) = (chh-c)/hh;
        X(i) = tmp;
    end
    matcompare(J,Jnum,'J',1e-6);

    % ipopt settings
    funcs.objective   = @(X) problem.objScale*sum(X(problem.iu).^2);
    funcs.gradient    = @gradStanding;
    funcs.constraints = @conStanding;
    funcs.jacobian    = @conjacStanding;
    funcs.jacobianstructure = @(X) problem.Jpattern;

    % bounds
    xlb = [0 0.5 -0.5  -0.5 -0.5 -0.5  -0.5 -0.5 -0.5 zeros(1,9)]';
    xub = [0 1.5  0.5   0.5  0.5  0.5   0.5  0.5  0.5 zeros(1,9)]';
    if strcmp(pose,'flexed')
        % prescribed asymmetric knee flexion
        xlb([5 8]) = [-0.1 -0.2];
        xub([5 8]) = [-0.1 -0.2];

    end    
    ulb = -0.1*ones(6,1);
    uub =  0.1*ones(6,1);
    options.lb = [xlb ; ulb ; xlb ; ulb];
    options.ub = [xub ; uub ; xub ; uub];

    % bounds for constraints
    options.cl = zeros(model.nf,1);
    options.cu = zeros(model.nf,1);

    options.ipopt.print_level = 0;
	options.ipopt.hessian_approximation = 'limited-memory'; 
    options.tol = 1e-4;
    options.constr_viol_tol = 1e-5;

    % initial guess for X (states and controls)
    rng(0)
    X0 = 0.5*(options.lb+options.ub) + 0.01*(options.ub-options.lb).*randn(nX,1);
    X0 = 0.5*(options.lb+options.ub);

    % solve
    [X, info] = ipopt(X0,funcs,options);
    if (info.status ~= 0)
        error('Minimal effort standing was not solved.')
    else
        fprintf('%25s: solved by IPOPT in %d iterations, objective value %8.3f.\n', 'standing', info.iter, info.objective)
        % save the solution on a file
        x = X(1:model.nx);
        u = X(model.nx + (1:model.nu));
        save(['standing-' pose], 'x','u');
    end

    % show the solution
    figure
    model.visualize(model,x);
    axis([-0.8 0.8 -0.1 1.4])
    plot(xlim(), [0 0], 'k');  % draw ground as a black line

    title(['standing (' num2str(info.iter) ' iterations)'])
    for i = 1:model.nDofs
        fprintf('q%02d: %10.6f\n', i, x(i));
    end
    for i = 1:model.nu
        fprintf('u%02d: %10.6f\n', i, u(i));
    end

end
%=====================================================================
function g = gradStanding(X)
global problem
    g = zeros(size(X));
    g(problem.iu) = 2*problem.objScale*X(problem.iu);
end
%====================================================================
function c = conStanding(X)       
    global problem
    model = problem.model;
    conScale = problem.conScale;
    h = problem.h;

    x1 = X(1:model.nx);
    x2 = X(model.nx+model.nu + (1:model.nx));
    u2 = X(2*model.nx + model.nu + (1:model.nu));
    c = conScale * model.dynamics(model,x2,(x2-x1)/h,u2);
    if any(any(isnan(c)))
        keyboard
    end
end
%=====================================================================
function J = conjacStanding(X)
    global problem
    model = problem.model;
    conScale = problem.conScale;
    h = problem.h;

    ix1 = 1:model.nx;
    ix2 = model.nx+model.nu + (1:model.nx);
    iu2 = 2*model.nx + model.nu + (1:model.nu);
    [~, dfdx, dfdxd, dfdu] = model.dynamics(model,X(ix2),(X(ix2)-X(ix1))/h,X(iu2));
    J = problem.Jpattern;
    J(:,ix1) = -conScale * dfdxd/h;
    J(:,ix2) =  conScale * (dfdx + dfdxd/h);
    J(:,iu2) = conScale * dfdu;
    if any(any(isnan(J)))
        keyboard
    end
end
%======================================================================
function testSimulation(model)
% run a forward simulation

    fprintf('\nForward simulation of jumping...\n')

    % time step and duration
    h = 0.005;
    nSteps = 100;

    % initial condition is standing solution
    s = load('standing-flexed');
    x0 = s.x;

    % determine Jacobian sparsity pattern
    t = 10.0;  % make sure the controller is "on"
    Jpattern = conjacSim(rand(model.nx,1));

    % configure IPOPT, with constant objective (we are not optimizing)
    funcs.objective   = @(x) 1;
    funcs.gradient    = @(x) zeros(size(x));
    funcs.constraints = @conSim;
    funcs.jacobian    = @conjacSim;
    funcs.jacobianstructure = @() Jpattern;
    options.cl = zeros(model.nx,1);
    options.cu = zeros(model.nx,1);
    options.ipopt.print_level = 0;
	options.ipopt.hessian_approximation = 'limited-memory';

    % initialize the video file output
    v = VideoWriter('simulation', 'MPEG-4');
    open(v);

    % do the time steps
    figure()
    totalIterations = 0;
    t = 0.0;

    for istep = 1:nSteps
        % solve f(x,(x-x0)/h,u) = 0 for x      
        [x, info] = ipopt(x0,funcs,options);
        if (info.status ~= 0)
            fprintf('%25s: TIME STEP FAILED at t = %.4f\n', ' simulation', times(istep));
            keyboard
            return
        end
        totalIterations = totalIterations + info.iter;

        % prepare for next time step
        t = t + h;
        x0 = x;

        % display
        clf
        model.visualize(model,x);
        title(['simulation -- t=' num2str(t,'%.3f')])
        axis([-0.8 1.0 -0.1 1.5]);
        drawnow;

        % save on video file
        frame = getframe(gcf);
        writeVideo(v,frame);
    end
    close(v);
    fprintf('%25s: completed, iterations per time step = %.2f\n', ' simulation', totalIterations/nSteps)    

    function f = conSim(x)
        u = controller(t,x);
        f = model.dynamics(model,x,(x-x0)/h,u);
    end
    function J = conjacSim(x)
        [u, du_dx] = controller(t,x);
        [~,df_dx,df_dxd,df_du] = model.dynamics(model,x,(x-x0)/h,u);
        J = df_dx + df_dxd/h + df_du * du_dx;
    end
    function [u, du_dx] = controller(t,x)
        % model is passive until t=0.3
        if (t < 0.3)
            u = zeros(model.nu,1);
            if (nargout > 1)
                du_dx = spalloc(model.nu,model.nx,0);
            end
        else
            % PD control
            Kp = 0.5;    % kN/rad
            Kd = 0.005;   % kNm/(rad/s)
            u = -Kp*x(4:9) - Kd*x(13:end);
            if (nargout > 1)
                du_dx = spalloc(model.nu,model.nx,12);
                du_dx(:,4:9) = -Kp*speye(6);
                du_dx(:,13:end) = -Kd*speye(6);
            end
        end
    end
end
