
boxdim = 1;
gridsize = 0.05;
Point(1) = {0.0,0.0,0.0,gridsize};
Point(2) = {boxdim,0.0,0.0,gridsize};
Point(3) = {boxdim,boxdim,0.0,gridsize};
Point(4) = {0.0,boxdim,0.0,gridsize};
Line(7) = {1,2};
Line(8) = {2,3};
Line(9) = {3,4};
Line(10) = {4,1};
Line Loop(14) = {7,8,9,10};
Plane Surface(16) = 14;
                    
                    
                    

Transfinite Line{7,8,9,10} = boxdim/gridsize;
Transfinite Surface{16};
