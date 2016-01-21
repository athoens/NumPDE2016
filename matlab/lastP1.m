function phi = lastP1(f, p, n)
%
% computes the element load vector
%
% input:
% f - function handle to source term
% p - 3x2-matrix of the coordinates of the triangle nodes
% n - order of the numerical integration rule (1 <= n <= 5)
%
% output:
% phi - element load vector (3x1-matrix)

% only pass admissible integration order n
if nargin == 2
    n = 1;
else
    n = max(1,min(5,floor(n)));
end

% read quadrature points
[x,w] = gaussTriangle(n);
x = (x + 1)/2;
w = w/4;

% number of quadrature points
k = size(x,2);

% vertices of the triangle
P0 = p(1,:)';
P1 = p(2,:)';
P2 = p(3,:)';

% Jacobian of the element map
Fk = [P1-P0, P2-P0];
detFk = det(Fk);

% numerical integration
phi = zeros(3,1);
for i = 1:k
    y = P0 + Fk*x(:,i);
    phi(1) = phi(1) + detFk * w(i) * f(y) * (1-x(1,i)-x(2,i));
    phi(2) = phi(2) + detFk * w(i) * f(y) * x(1,i);
    phi(3) = phi(3) + detFk * w(i) * f(y) * x(2,i);
end