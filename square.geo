// Points //
Point(1) = {-1, -1, 0};
Point(2) = {-1,  0, 0};
Point(3) = {-1,  1, 0};
Point(4) = { 1,  1, 0};
Point(5) = { 1,  0, 0};
Point(6) = { 1, -1, 0};
Point(7) = { 0,  0, 0};

// Lines //
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line(7) = {2, 7};
Line(8) = {7, 5};

// Surfaces //
Line Loop(14) = {2, 3, 4, -8, -7};
Plane Surface(15) = {14};
Line Loop(16) = {1, 7, 8, 5, 6};
Plane Surface(17) = {16};

// Attributes //
//Physical Point(18) = {2, 5};   // singular points
Physical Line(19) = {2, 3, 4}; // Neumann b.c.
Physical Line(20) = {1, 6, 5}; // Dirichlet b.c.

Physical Surface(201) = {15, 17};
