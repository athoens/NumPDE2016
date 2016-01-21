function b = rieszTriangleP2(p, t, eIndex, f, n)

% Returns the load vector for 
%   int_Omega f v dx 
% for quadratic FEM on triangles
%
% input:
% p      - Nx2-matrix with coordinates of the nodes
% t      - Mx3-matrix with indices of nodes of the triangles
% eIndex - NxN-matrix with indices of edges
% f      - function handle to source term
% n      - order of numerical quadrature
%
% output:
% b - (N+E)x1 vector

% if not passed as input parameter set order of numerical quadrature to 1
if nargin < 5
    n = 1;
end

% number of nodes
N = size(p,1);

% number of triangles
M = size(t,1);

% number of edges
E = full(max(max(eIndex)));

% assemble load vector
b = zeros(N+E,1);
for i = 1:M
    b_index = [t(i,:),...
               N+eIndex(t(i,2),t(i,3))...
               N+eIndex(t(i,3),t(i,1))...
               N+eIndex(t(i,1),t(i,2))];
    b(b_index,1) = b(b_index,1) + lastP2(f,p(t(i,:),:),n);
end