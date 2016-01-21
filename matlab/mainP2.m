function [u,A,b]=mainP2(h0,n)

% function performing all steps of exercise 5f
%
% input:
% h0 - [optional] maximum mesh width
% n  - [optional] order of numerical integration
%
% output:
% u  - solution coefficient vector
% A  - stiffness matrix
% b  - load vector


% include m-files of 7th series
includeSeries7;

% parameters
if ~exist('h0','var')
    h0 = 0.1; % if not passed as input parameter set maximum mesh width
end
if ~exist('n','var')
    n  = 5;   % order of numerical quadrature
end

% function handle to source term
f = @(x) (2*pi^2*sin(pi*x(1))*sin(pi*x(2)));

% create mesh
[p,t] = meshSquare(1,h0,[0.5,0.5]);

% read number of nodes
N = size(p,1);

% read boundary edges and identify boundary nodes and inner nodes
[e,eIndex,boundaryNodes,boundaryEdges] = edgeMatrix(p,t);
E = size(e,1);

% identify inner nodes and edges
innerNodes = setdiff(1:N,boundaryNodes);
innerEdges = setdiff(1:E,boundaryEdges);

% write inner DoFs
innerDofs = [innerNodes,N+innerEdges];

% write stiffness matrix and load vector;
A = stiffnessP2(p,t,eIndex);
b = rieszTriangleP2(p,t,eIndex,f,n);

% solve problem
u = zeros(N+E,1);
u(innerDofs) = A(innerDofs,innerDofs)\b(innerDofs);

% check if stiffness matrix and rhs corresponding to the vertices coincide
% with P1 stiffness matrix and rhs
%A_P1 = stiffnessP1(p,t);
%b_P1 = rieszTriangleP1(p,t,f,n);
%fprintf('\n||A - A_P1|| = %d\n||b - b_P1|| = %d\n\n',...
%    norm(full(A(1:N,1:N)-A_P1)),...
%    norm(b(1:N,1)-b_P1));

% analytical solution
U = sin(pi*p(:,1)).*sin(pi*p(:,2));

% plot solution
figure(11);clf;
trimesh(t,p(:,1),p(:,2),u(1:N));
title('numerical solution');
figure(12);clf;
trimesh(t,p(:,1),p(:,2),U);
title('analytical solution');
figure(13);clf;
trimesh(t,p(:,1),p(:,2),abs(U-u(1:N)));
title('discretization error');
figure(14);clf;
trimesh(t,p(:,1),p(:,2),b(1:N));
title('rhs');