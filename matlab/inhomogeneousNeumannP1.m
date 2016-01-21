function [N,lambda]=inhomogeneousNeumannP1(h0)

% function performing all steps of exercise 2g
%
% input:
% h0 - [optional] maximum mesh width
%
% output:
% N      - number of nodes
% lambda - Lagrange multiplier


% parameters
if nargin == 0
    h0 = 0.1; % if not passed as input parameter set maximum mesh width
end
n  = 1;   % order of numerical quadrature

% include m-files of 7th series
includeSeries7;

% function handle to f and g
r = @(x) (sqrt(x(1)^2+x(2)^2));
f = @(x) (-x(1)*(3*pi*cos(pi*r(x))/r(x)-pi^2*sin(pi*r(x))));
g = @(x) (pi*x(1));

% create mesh
[p,t] = meshRing(2,1,200,h0,[0,0]);

% read dimension
N = size(p,1);

% write stiffness matrix and load vector;
A = stiffnessP1(p,t);
b = rieszTriangleP1(p,t,f,n) + neumannP1(p,boundaryEdges(t),g,n);
m = rieszTriangleP1(p,t,@(x) 1,n);

% solve problem
U      = [A,m;m',0]\[b;0];
u      = U(1:N,1);
lambda = U(N+1,1);

% analytical solution
U = p(:,1).*sin(pi*sqrt(p(:,1).^2+p(:,2).^2));

% plot solution
figure(4);clf;
trimesh(t,p(:,1),p(:,2),u);
title('numerical solution');
figure(5);clf;
trimesh(t,p(:,1),p(:,2),U);
title('analytical solution');
figure(6);clf;
trimesh(t,p(:,1),p(:,2),abs(U-u));
title('discretization error');