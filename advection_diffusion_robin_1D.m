% advection_diffusion_robin_1D.m
%
% Solves the 1D advection-diffusion equation
%
%       u_t + a * u_x = nu * u_xx,      x in [west, east]
%
% using the MOLE library's mimetic divergence, gradient and interpolation
% operators (jcorbino-mole-83ffb7f/mole_MATLAB), with Robin (mixed)
% boundary conditions at BOTH ends:
%
%       ar*u(west) - br*u_x(west) = g_west   (inflow: partially-resistive supply)
%       ar*u(east) + br*u_x(east) = g_east   (outflow: convective/radiative loss)
%
% Pure advection has no natural Robin condition (it only needs one BC, at
% the inflow boundary), so this uses the advection-DIFFUSION equation
% instead, which is second-order and genuinely needs -- and supports --
% a Robin condition at each end, the same way MOLE's own elliptic1D.m
% example uses robinBC for the (steady) Poisson equation.
%
% The animation keeps looping in time (replaying t = 0 -> tFinal) until
% you close the figure window.

clc
close all

% --- Make the MOLE MATLAB library visible, regardless of the current
%     working directory, since this script lives outside the library. ---
thisDir = fileparts(mfilename('fullpath'));
addpath(fullfile(thisDir, 'jcorbino-mole-83ffb7f', 'mole_MATLAB'));

% --- Physical parameters ---
a  = 1;      % Advection velocity
nu = 0.02;   % Diffusivity
west = 0;    % Domain limits
east = 1;

% --- Discretization ---
k  = 4;                     % Mimetic operator order of accuracy (2, 4, 6, or 8)
m  = 60;                    % Number of cells (must satisfy m >= 2*k+1)
dx = (east - west) / m;

tFinal = 1.5;                % Simulation end time (the front crosses the
                              % domain 1.5 times, so the outflow Robin BC
                              % actually gets exercised)
nSteps = 150;
dt     = tFinal / nSteps;

% 1D staggered grid: two boundary nodes plus m cell centers
grid = [west, west+dx/2 : dx : east-dx/2, east]';

% --- Mimetic operators ---
D = div(k, m, dx);       % (m+2) x (m+1)
G = grad(k, m, dx);      % (m+1) x (m+2)
Ip = interpol(m, 0.5);   % (m+1) x (m+2), 2nd-order centered

% Spatial operator: -a*d/dx + nu*d2/dx2. Like D and G on their own, S has
% zero boundary rows (rows 1 and m+2) -- they're placeholders, meant to be
% overwritten by a BC operator.
S = -a*(D*Ip) + nu*(D*G);

% --- Robin BC operator (fills exactly those zero boundary rows) ---
ar = 1;      % Dirichlet-type coefficient
br = 0.1;    % Neumann-type coefficient
BC = robinBC(k, m, dx, ar, br);

g_west = 1;   % ar*u - br*u_x = g_west   at x = west (inflow)
g_east = 0;   % ar*u + br*u_x = g_east   at x = east (outflow)

% --- Backward Euler system matrix, with the boundary rows swapped out for
%     the (time-invariant) Robin condition: (I - dt*S) everywhere except
%     the two boundary rows, which instead enforce the Robin BC exactly. ---
M = speye(m+2) - dt*S;
M(1, :)   = BC(1, :);
M(end, :) = BC(end, :);

U0 = zeros(m+2, 1);   % Initial condition: quiescent domain

fig = figure;
while ishandle(fig)
    U = U0;

    for n = 0:nSteps
        if ~ishandle(fig)
            break
        end

        t = n * dt;

        plot(grid, U, 'o-')
        axis([west east -0.2 1.2])
        xlabel('x')
        ylabel('u(x,t)')
        title(sprintf('Advection-diffusion, Robin BC \t t = %.3f', t))
        drawnow

        if n < nSteps
            RHS = U;
            RHS(1)   = g_west;
            RHS(end) = g_east;
            U = M \ RHS;
        end
    end
end
