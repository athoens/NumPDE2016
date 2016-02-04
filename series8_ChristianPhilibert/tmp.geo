
// HA3 structure
h0 = 0.0625;
a= 1;
Point(1) = {0, 0, 0, h0};
Point(2) = {a, 0, 0, h0};
Point(3) = {a, a, 0, h0};
Point(4) = {0, a, 0, h0};
Line(1) = {3, 4};
Line(2) = {4, 1};
Line(3) = {1, 2};
Line(4) = {2, 3};
Line Loop(6) = {1, 2, 3, 4};
Plane Surface(6) = {6};
// make it regular
Transfinite Surface {6};
// do the meshing
Mesh 2;
// save
Save "tmp.msh";
