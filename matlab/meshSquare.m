function [p,t]=meshSquare(l,h0,center)
%
% creates a mesh for a square with side length l and maximum mesh width h0
%
% input:
% l      - side length of the square
% h0     - maximum mesh width
% center - [optional] center of the square (if left empty, center is set to
%          origin)
%
% output:
% p - array of nodal coordinates
% t - array of triangles as indicies into p, defined with a 
%      counter-clockwise node ordering

if nargin<3
    center=[0,0];
end

hdata.hmax = h0;

node = [ center(1)-l/2, center(2)-l/2;...
         center(1)+l/2, center(2)-l/2;...
         center(1)+l/2, center(2)+l/2;...
         center(1)-l/2, center(2)+l/2];

options.output = false;
warning off;
[p,t]=mesh2d(node,[],hdata,options);
warning on;