function [N,H,N2H,E]=convergenceP1(h0)

% computes the energy norm of the discretization error
%
% input:
% h0  - [optional] maximum mesh width
%
% output:
% N   - vector of number of nodes
% H   - vector of maximum mesh width as passed to mesh2d
% N2H - vector of maximum mesh width as computed from number of nodes
% E   - vector of discretization error in energy norm


% include m-files of 7th series
includeSeries7;

% parameters
if nargin == 0
    h0 = 0.1; % if not passed as input parameter set minimum mesh width
end
n  = 5;   % order of numerical quadrature

% function handle to source term
f = @(x) (2*pi^2*sin(pi*x(1))*sin(pi*x(2)));

% vectors for number of nodes, mesh width and energy norm
L   = 7;
N   = zeros(L,1);
H   = zeros(L,1);
E   = zeros(L,1);
N2H = zeros(L,1);

% number of rows
nr = 0;

% loop through maximum mesh widths
for l=0:L-1
    
    % maximum mesh width
    hl      = h0*2^(-l/2);
    H(nr+1) = hl;

    % create mesh
    [p,t] = meshSquare(1,hl,[0.5,0.5]);

    % read dimension
    N(nr+1)   = size(p,1);
    N2H(nr+1) = sqrt(2)/(sqrt(N(nr+1))-1);
    
    % check if number of nodes has increased
    if (l == 0) || (N(nr+1) > N(nr))

        % increment number of rows
        nr = nr + 1;
        
        % read boundary edges and identify boundary nodes and inner nodes
        boundaryNodes = unique(boundaryEdges(t));
        innerNodes    = setdiff(1:N(nr),boundaryNodes);

        % write stiffness matrix and load vector;
        A = stiffnessP1(p,t);
        b = rieszTriangleP1(p,t,f,n);

        % solve problem
        u = zeros(N(nr),1);
        u(innerNodes) = A(innerNodes,innerNodes)\b(innerNodes);

        % compute energy norm of discretization error
        E(nr) = sqrt(pi^2/2 - dot(b(innerNodes),u(innerNodes)));
    end
end

% delete empty rows
N   = N(1:nr,1);
H   = H(1:nr,1);
E   = E(1:nr,1);
N2H = N2H(1:nr,1);

% loglog plot
figure(1);clf;
loglog(H,E,'k-',H,H,'r-');grid;
legend('energy norm of error', 'expected convergence');
ylabel('energy norm');
xlabel('maximum mesh width as passed to Mesh2D');
figure(2);clf;
loglog(N,E,'k-',N,N.^(-1/2),'r-');grid;
legend('energy norm of error', 'expected convergence');
ylabel('energy norm');
xlabel('number of nodes');
figure(3);clf;
loglog(N2H,E,'k-',N2H,N2H,'r-');grid;
legend('energy norm of error', 'expected convergence');
ylabel('energy norm');
xlabel('maximum mesh width');

% print convergence rate
fprintf('\nConvergence Rate of Energy Norm of Discretization Error\n\n');
fprintf('with respect to maximum mesh width as passed to Mesh2D:\n');
fprintf('%d\n',diff(log(E))./diff(log(H)));
fprintf('\n');
fprintf('with respect to number of nodes:\n');
fprintf('%d\n',diff(log(E))./diff(log(N)));
fprintf('\n');
fprintf('with respect to actual maximum mesh width:\n');
fprintf('%d\n',diff(log(E))./diff(log(N2H)));
fprintf('\n');