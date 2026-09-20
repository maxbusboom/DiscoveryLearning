% heat_equation_1D.m
%
% Solves the 1D heat equation
%
%       u_t = alpha * u_xx,      x in [west, east]
%
% with homogeneous Dirichlet boundary conditions, using the mimetic
% Laplacian operator from the MOLE library (jcorbino-mole-83ffb7f/mole_MATLAB).
%
% The initial condition u(x,0) = sin(pi*x) is chosen because the exact
% solution is known in closed form:
%
%       u(x,t) = exp(-alpha*pi^2*t) * sin(pi*x)
%
% so the mimetic solution can be checked directly against it.
%
% Time integration uses backward Euler, which is unconditionally stable,
% so dt is not restricted by a von Neumann criterion the way an explicit
% scheme would be.

clc
close all

% --- Make the MOLE MATLAB library visible, regardless of the current
%     working directory, since this script lives outside the library. ---
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'jcorbino-mole-83ffb7f', 'mole_MATLAB'));

% --- Physical parameters ---
alpha = 0.1;   % Thermal diffusivity
west  = 0;     % Domain limits
east  = 1;

% --- Discretization ---
k  = 4;                     % Mimetic operator order of accuracy (2, 4, 6, or 8)
m  = 40;                    % Number of cells (must satisfy m >= 2*k+1)
dx = (east - west) / m;

tFinal = 0.5;                % Simulation end time
nSteps = 200;
dt     = tFinal / nSteps;

% 1D staggered grid: two boundary nodes plus m cell centers
grid = [west, west+dx/2 : dx : east-dx/2, east]';

% --- Mimetic Laplacian operator ---
L = lap(k, m, dx);   % (m+2) x (m+2) sparse operator

% --- Initial condition (already zero at both boundaries) ---
U = sin(pi * grid);

% --- Backward Euler system: (I - alpha*dt*L) * U^{n+1} = U^n ---
% The boundary rows of L are identically zero (they are placeholders in
% the standard, non-periodic operator), so the corresponding rows of A
% reduce to identity rows. That freezes U(1) and U(end) at their initial
% values, which enforces the homogeneous Dirichlet BC for all time.
A = speye(m+2) - alpha*dt*L;

% The animation keeps looping in time (replaying t = 0 -> tFinal) until
% you close the figure window.
U0 = U;

fig = figure;
while ishandle(fig)
    U = U0;

    for n = 0:nSteps
        if ~ishandle(fig)
            break
        end

        t = n * dt;
        Uexact = exp(-alpha*pi^2*t) * sin(pi*grid);

        plot(grid, U, 'o-', grid, Uexact, 'k--')
        axis([west east -0.1 1.1])
        legend('Mimetic (MOLE)', 'Analytical', 'Location', 'northeast')
        xlabel('x')
        ylabel('u(x,t)')
        title(sprintf('1D heat equation \t t = %.3f', t))
        drawnow

        if n < nSteps
            U = A \ U;   % Solve the linear system (unconditionally stable)
        end
    end

    if ishandle(fig)
        err = max(abs(U - Uexact));
        fprintf('Max error at t = %.3f (k = %d, m = %d):  %.3e\n', tFinal, k, m, err);
    end
end
