function b = rieszTriangleP1(p, t, f, n)

% Returns the load vector for 
%   int_Omega f v dx 
% for linear FEM on triangles
%
% input:
% p - Nx2-matrix with coordinates of the nodes
% t - Mx3-matrix with indices of nodes of the triangles
% f - function handle to source term
% n - order of numerical quadrature
%
% output:
% b - Nx1 vector

% if not passed as input parameter set order of numerical quadrature to 1
if nargin == 3
    n = 1;
end

% number of nodes
N = size(p,1);

% number of triangles
M = size(t,1);

% assemble load vector
b = zeros(N,1);
for i=1:M
    b(t(i,:),1) = b(t(i,:),1) + lastP1(f,p(t(i,:),:),n);
end