function Ak=laplaceP2(p)
%
% computes the element stiffness matrix
%
% input:
% p - 3x2-matrix of the coordinates of the triangle nodes
%
% output:
% Ak - 6x6 element stiffness matrix

% vertices of the triangle
P0 = p(1,:)';
P1 = p(2,:)';
P2 = p(3,:)';

% Jacobian of the element map and its determinant
Fk = [P1-P0, P2-P0];
detFk = det(Fk); % = 2*|K|

% coordinate difference matrix
Dk = [[ P1(2)-P2(2), P2(2)-P0(2), P0(2)-P1(2) ];...
      [ P2(1)-P1(1), P0(1)-P2(1), P1(1)-P0(1) ]];

% gradient matrix multiplied with |K|
Gk = (1/2) * (1/detFk) * transpose(Dk)*Dk;
diagGk = diag(Gk);

% "transformation" matrices
T1 = ones(3,3)-eye(3);
T2 = [[0,0,1];[1,0,0];[0,1,0]];

% node and edge indices
nodes = 1:3;
edges = 4:6;

% element stiffness matrix
Ak(nodes,nodes) = Gk;
Ak(nodes,edges) = (1/3) * Gk * T1;
Ak(edges,nodes) = transpose(Ak(nodes,edges));
Ak(edges,edges) = (1/12) * (transpose(T1) * Gk * T1 ...
                            + Gk ...
                            - diag(diagGk) ...
                            + diag(T2*diagGk) ...
                            + diag(T2*T2*diagGk));