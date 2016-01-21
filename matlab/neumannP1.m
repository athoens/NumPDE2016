function b = neumannP1(p, boundaryEdges, g, n)

% Returns the vector related to the boundary data
%   int_dOmega g v ds(x) 
% for linear FEM in straight intervals
%
% input:
% p             - Nx2-matrix with coordinates of the nodes
% boundaryEdges - Bx2-matrix with indices of nodes of boundary edges
% g             - function handle to function g
% n             - order of numerical quadrature
%
% output:
% b - Nx1 vector

% if not passed as input parameter set order of numerical quadrature to 1
if nargin == 3
    n = 1;
end

% number of nodes
N = size(p,1);

% number of boundaryEdges
B = size(boundaryEdges,1);

% assemble vector
b = zeros(N,1);
for i=1:B
    b(boundaryEdges(i,:),1) = b(boundaryEdges(i,:),1) ...
        + lastNeumannP1(g,p(boundaryEdges(i,:),:),n);
end