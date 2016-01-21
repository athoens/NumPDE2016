function A = stiffnessP1(p, t)

% Returns the stiffness matrix A for 
%   int_Omega grad u . grad v dx
% for linear FEM on triangles
%
% input:
% p - Nx2 matrix with coordinates of the nodes
% t - Mx3 matrix with indices of nodes of the triangles
%
% output:
% A - NxN stiffness matrix in sparse format

% number of nodes
N = size(p,1);

% number triangles
M = size(t,1);

% assemble stiffness matrix
A = sparse(N,N);
for i=1:M
    A(t(i,:),t(i,:)) = A(t(i,:),t(i,:)) + LaplaceP1(p(t(i,:),:));
end