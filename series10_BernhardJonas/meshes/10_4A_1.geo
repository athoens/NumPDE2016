Point(1) = {-1.0, -1.0, 0, 0.3535533905932738};
Point(2) = { 1.0, -1.0, 0, 0.3535533905932738};
Point(3) = { 1.0,  1.0, 0, 0.3535533905932738};
Point(4) = {-1.0,  1.0, 0, 0.3535533905932738};
Point(5) = {-1.0,  0, 0, 0.06250000000000001};
Point(7) = { 1.0,  0, 0, 0.06250000000000001};
Point(6) = {0, 0, 0, 0.3535533905932738};
Line(1) = {1, 2};
Line(2) = {2, 7};
Line(3) = {7, 6};
Line(4) = {6, 5};
Line(5) = {5, 1};
Line(6) = {7, 3};
Line(7) = {3, 4};
Line(8) = {4, 5};
Line Loop(1) = {1, 2, 3, 4, 5};
Line Loop(2) = {-4,-3, 6, 7, 8};
Plane Surface(1) = {1};
Plane Surface(2) = {2};
