function [p,t]=meshRing(r1,r2,n,h0,center)
%
% creates a mesh for a ring with outer radius r1 and inner radius r2
%
% input:
% r1     - outer/inner radius of the ring
% r2     - outer/inner radius of the ring
% n      - number of nodes on the polygon defining the "circle" 
% h0     - maximum mesh width
% center - [optional] center of the circle (if left empty, center is set to
%          origin)
%
% output:
% p - array of nodal coordinates
% t - array of triangles as indicies into p, defined with a 
%      counter-clockwise node ordering

if nargin<5
    center=[0,0];
end

hdata.hmax = h0;

theta = transpose(linspace(-pi,pi,n+1));
theta = theta(1:n,1);
node1 = [center(1)+r1*cos(theta),center(2)+r1*sin(theta)];
node2 = [center(1)+r2*cos(theta),center(2)+r2*sin(theta)];
node  = [node1;node2];


cnect = [...
         [ (1:n)',     [(2:n)';       1]];...
         [ (n+1:2*n)', [(n+2:2*n)'; n+1]]...
        ];

options.output = false;
warning off;
[p,t]=mesh2d(node,cnect,hdata,options);
warning on;