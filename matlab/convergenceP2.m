function [N,H,EP1,EP2]=convergenceP2(h0)

% computes the energy norm of the discretization error
%
% input:
% h0  - [optional] maximum mesh width
%
% output:
% N   - vector of number of nodes
% H   - vector of maximum mesh width, i.e. H = N^(-1/2)
% EP1 - vector of discretization error in energy norm (p=1)
% EP2 - vector of discretization error in energy norm (p=2)


% include m-files of 7th series
includeSeries7;

% parameters
if nargin == 0
    h0 = 0.1; % if not passed as input parameter set minimum mesh width
end
n  = 5;   % order of numerical quadrature

% function handle to source term
f = @(x) (2*pi^2*sin(pi*x(1))*sin(pi*x(2)));

% vectors for number of nodes and energy norm of discretization error
L   = 7;
N   = zeros(L,1);
EP1 = zeros(L,1);
EP2 = zeros(L,1);

% number of rows
nr = 0;

% loop through maximum mesh widths
for l=0:L-1
    
    % maximum mesh width
    hl      = h0*2^(-l/2);
    
    % create mesh
    [p,t] = meshSquare(1,hl,[0.5,0.5]);

    % read dimension
    N(nr+1) = size(p,1);
    
    % check if number of nodes has increased
    if (l == 0) || (N(nr+1) > N(nr))

        % increment number of rows
        nr = nr + 1;
        
        % read boundary edges and identify boundary nodes and inner nodes
        [e,eIndex,boundaryNodes,boundaryEdges] = edgeMatrix(p,t);
        
        % identify inner nodes and edges
        innerNodes = setdiff(1:N(nr),    boundaryNodes);
        innerEdges = setdiff(1:size(e,1),boundaryEdges);

        % write inner DoFs
        innerDofsP1 = innerNodes;
        innerDofsP2 = [innerNodes,N(nr)+innerEdges];
        
        % write stiffness matrix and load vector
        AP1 = stiffnessP1(p,t);
        bP1 = rieszTriangleP1(p,t,f,n);
        AP2 = stiffnessP2(p,t,eIndex);
        bP2 = rieszTriangleP2(p,t,eIndex,f,n);
        
        % solve problem
        uP1 = zeros(N(nr),1);
        uP1(innerDofsP1) = AP1(innerDofsP1,innerDofsP1)\bP1(innerDofsP1);
        uP2 = zeros(N(nr)+size(e,1),1);
        uP2(innerDofsP2) = AP2(innerDofsP2,innerDofsP2)\bP2(innerDofsP2);
        
        % compute energy norm of discretization error
        EP1(nr) = sqrt(pi^2/2 - dot(bP1,uP1));
        EP2(nr) = sqrt(pi^2/2 - dot(bP2,uP2));
    end
end

% delete empty rows
N   = N(1:nr,1);
H   = 1./(sqrt(N)-1);
EP1 = EP1(1:nr,1);
EP2 = EP2(1:nr,1);


% loglog plot
figure(11);clf;
loglog(H,EP1,'b-',H,H,'b--',H,EP2,'r-',H,H.*H,'r--');grid;
legend('energy norm of error (p=1)', 'expected convergence (p=1)',...
    'energy norm of error (p=2)', 'expected convergence (p=2)');
ylabel('energy norm');
xlabel('maximum mesh width');

% print convergence rate
fprintf('\nConvergence Rate of Energy Norm of Discretization Error\n\n');
fprintf('p = 1:\n');
fprintf('%d\n',diff(log(EP1))./diff(log(N)));
fprintf('\n');
fprintf('p = 2:\n');
fprintf('%d\n',diff(log(EP2))./diff(log(N)));
fprintf('\n');