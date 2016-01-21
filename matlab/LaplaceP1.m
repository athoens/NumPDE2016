function Ak=LaplaceP1(p)
%
% computes the element stiffness matrix
%
% input:
% p - 3x2-matrix of the coordinates of the triangle nodes
%
% output:
% Ak - element stiffness matrix

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

% element stiffness matrix
Ak = (1/2) * (1/detFk) * transpose(Dk)*Dk;