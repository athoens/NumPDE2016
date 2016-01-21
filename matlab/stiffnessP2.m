function A = stiffnessP2(p, t, eIndex)

% Returns the stiffness matrix A for 
%   int_Omega grad u . grad v dx
% for quadratic FEM on triangles
%
% input:
% p      - Nx2 matrix with coordinates of the nodes
% t      - Mx3 matrix with indices of nodes of the triangles
% eIndex - NxN-matrix with indices of edges
%
% output:
% A - (N+E)x(N+E) stiffness matrix in sparse format

% number of nodes
N = size(p,1);

% number triangles
M = size(t,1);

% number of edges
E = full(max(max(eIndex)));

% assemble stiffness matrix
A = sparse(N+E,N+E);
for i=1:M
    b_index = [t(i,:),...
               N+eIndex(t(i,2),t(i,3))...
               N+eIndex(t(i,3),t(i,1))...
               N+eIndex(t(i,1),t(i,2))];
    A(b_index,b_index) = A(b_index,b_index) + laplaceP2(p(t(i,:),:));
end