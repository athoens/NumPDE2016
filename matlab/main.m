function [u,A,b]=main(h0)

% function performing all steps of exercise 3f
%
% input:
% h0 - [optional] maximum mesh width
%
% output:
% u  - solution coefficient vector
% A  - stiffness matrix
% b  - load vector


% parameters
if nargin == 0
    h0 = 0.1; % if not passed as input parameter set maximum mesh width
end
n  = 3;   % order of numerical quadrature

% function handle to source term
f = @(x) (2*pi^2*sin(pi*x(1))*sin(pi*x(2)));
%f = @(x) (1);
%f = @(x)(2*pi^2 * sin(pi*x(1))*sin(pi*x(2)) * (x(2)^2) * exp(-x(1)));

% create mesh
[p,t] = meshSquare(1,h0,[0.5,0.5]);

% read dimension
N = size(p,1);

% read boundary edges and identify boundary nodes and inner nodes
boundaryNodes = unique(boundaryEdges(t));
innerNodes    = setdiff(1:N,boundaryNodes);

% write stiffness matrix and load vector;
A = stiffnessP1(p,t);
b = rieszTriangleP1(p,t,f,n);

% solve problem
u = zeros(N,1);
u(innerNodes) = A(innerNodes,innerNodes)\b(innerNodes);

% analytical solution
U = sin(pi*p(:,1)).*sin(pi*p(:,2));

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
figure(4);clf;
trimesh(t,p(:,1),p(:,2),b);
title('rhs');