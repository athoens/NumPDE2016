function bK = lastNeumannP1(g, p, n)
%
% computes the element load vector related to the 
%
% input:
% g - function handle to function g
% p - 2x2-matrix of the two coordinates of the boundary interval
% n - order of the numerical integration rule
%
% output:
% bK - element Neumann load vector (2x1-matrix)

% only pass admissible integration order n
if nargin == 2
    n = 1;
else
    n = floor(n);
end

% nodes of the interval
P0 = p(1,:)';
P1 = p(2,:)';

% length of interval
L = norm(P0-P1,2);

% read quadrature points and weights
[x,w] = gauleg(n);
x     = (x + 1)/2; % transform quadrature points to interval [0,1]
w     = w/2;       % scale weights according to transformation

% numerical integration
bK = zeros(2,1);
for i = 1:n
    % transform quadrature points to interval [P0,P1]
    y = P0 + x(i)*(P1-P0);
    % add weight w(i) multiplied with function g at y multiplied with
    % element basis function multiplied with length L of interval
    bK(1) = bK(1) + L * w(i) * g(y) * (1-x(i));
    bK(2) = bK(2) + L * w(i) * g(y) * x(i);
end