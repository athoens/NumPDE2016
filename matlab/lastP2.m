function phi = lastP2(f,p,n)
%
% computes the element load vector
%
% input:
% f - function handle to source term
% p - 3x2-matrix of the coordinates of the triangle nodes
% n - order of the numerical integration rule (1 <= n <= 5)
%
% output:
% phi - element load vector (6x1-matrix)

% only pass admissible integration order n
if nargin < 3
    n = 1;
else
    n = floor(n);
end

% read quadrature points, transform them to integral [0,1] and adjust
% weights
[eta,w] = gauleg(n);
eta     = (eta + 1)/2;

% vertices of the triangle
P1 = p(1,:)';
P2 = p(2,:)';
P3 = p(3,:)';

% Jacobian of the element map
Fk = [P2-P1, P3-P1];
detFk = det(Fk);

% barycentric coordinates
lambda1 = @(x) ((1/detFk) * dot(x-P2,[P2(2)-P3(2);P3(1)-P2(1)]));
lambda2 = @(x) ((1/detFk) * dot(x-P3,[P3(2)-P1(2);P1(1)-P3(1)]));
lambda3 = @(x) ((1/detFk) * dot(x-P1,[P1(2)-P2(2);P2(1)-P1(1)]));

% shape functions
N0 = @(x) lambda1(x);
N1 = @(x) lambda2(x);
N2 = @(x) lambda3(x);
N3 = @(x) lambda2(x)*lambda3(x);
N4 = @(x) lambda1(x)*lambda3(x);
N5 = @(x) lambda1(x)*lambda2(x);

% numerical integration
phi = zeros(6,1);
for i = 1:n
    for j = 1:n
        % Duffy transformation
        xi = [eta(i)*(1-eta(j)); eta(j)];
        % element map
        x = P1 + Fk*xi;
        % Jacobian of the Duffy transformation
        detD = 1 - eta(j);
        % integrate
        phi(1) = phi(1) + (1/4) * detFk * detD * w(i) * w(j) * f(x) * N0(x);
        phi(2) = phi(2) + (1/4) * detFk * detD * w(i) * w(j) * f(x) * N1(x);
        phi(3) = phi(3) + (1/4) * detFk * detD * w(i) * w(j) * f(x) * N2(x);
        phi(4) = phi(4) + (1/4) * detFk * detD * w(i) * w(j) * f(x) * N3(x);
        phi(5) = phi(5) + (1/4) * detFk * detD * w(i) * w(j) * f(x) * N4(x);
        phi(6) = phi(6) + (1/4) * detFk * detD * w(i) * w(j) * f(x) * N5(x);
    end
end