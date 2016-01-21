function [p,t]=meshCircle(r,n,h0,center)
%
% creates a mesh for a circle with radius r
%
% input:
% r      - radius of the circle
% n      - number of nodes on the polygon defining the "circle" 
% h0     - maximum mesh width
% center - [optional] center of the circle (if left empty, center is set to
%          origin)
%
% output:
% p - array of nodal coordinates
% t - array of triangles as indicies into p, defined with a 
%      counter-clockwise node ordering

if nargin<4
    center=[0,0];
end

hdata.hmax = h0;

theta = transpose(linspace(-pi,pi,n+1));
theta = theta(1:n,1);
node  = [center(1)+r*cos(theta),center(2)+r*sin(theta)];

options.output = false;
warning off;
[p,t]=mesh2d(node,[],hdata,options);
warning on;