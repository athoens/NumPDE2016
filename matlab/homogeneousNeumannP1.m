function [N,lambda]=homogeneousNeumannP1(h0)

% function performing all steps of exercise 2d
%
% input:
% h0 - [optional] maximum mesh width
%
% output:
% N      - number of nodes
% lambda - Lagrange multiplier


% parameters
if nargin == 0
    h0 = 0.1; % if not passed as input parameter set minimum mesh width
end
n  = 5;   % order of numerical quadrature

% include m-files of 7th series
includeSeries7;

% function handle to source term
f = @(x) (2*pi^2*cos(pi*x(1))*cos(pi*x(2)));

% create mesh
[p,t] = meshSquare(1,h0,[0.5,0.5]);

% read dimension
N = size(p,1);

% write stiffness matrix and load vector;
A = stiffnessP1(p,t);
b = rieszTriangleP1(p,t,f,n);
m = rieszTriangleP1(p,t,@(x) 1,n);

% solve problem
U      = [A,m;m',0]\[b;0];
u      = U(1:N,1);
lambda = U(N+1,1);

% analytical solution
U = cos(pi*p(:,1)).*cos(pi*p(:,2));

% plot solution
figure(1);clf;
trimesh(t,p(:,1),p(:,2),u);
title('numerical solution');
figure(2);clf;
trimesh(t,p(:,1),p(:,2),U);
title('analytical solution');
figure(3);clf;
trimesh(t,p(:,1),p(:,2),abs(U-u));
title('discretization error');