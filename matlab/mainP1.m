function mainP1(h0)

% function performing all steps of exercise 2.e) and 2.g)
%
% input:
% h0 - [optional] maximum mesh width

% if not passed as input parameter set maximum mesh width to 0.1
if nargin == 0
    h0 = 0.1;
end

% 2.d)
[~,lambda]=homogeneousNeumannP1(h0);
fprintf('\nhomgeneous Neumann problem:   lambda = %d\n',lambda);

% 2.g)
[~,lambda]=inhomogeneousNeumannP1(h0);
fprintf('\ninhomgeneous Neumann problem: lambda = %d\n\n',lambda);