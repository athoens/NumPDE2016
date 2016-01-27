#
# import other modules
#
import meshpy.triangle as triangle
import numpy as np
import numpy.linalg as la
from scipy.sparse import lil_matrix
import FEM
import math as m
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt

# CREATEBGMESH  creates a background mesh for gmsh for square domains
#
# createbgMesh(filename, intervX, intervY, meshwidth)
# e.g. createbgMesh('bgmesh.pos', [-1,1], [-1,1], 0.5, @meshwidth_graded, 101)

def createbgMesh(filename, intervX, intervY, meshwidth, N=25):

  minx = min(intervX);
  maxx = max(intervX);
  miny = min(intervY);
  maxy = max(intervY);
  
  x = np.linspace(minx, maxx, N);
  y = np.linspace(miny, maxy, N);
  #print x.shape
  [X,Y] = np.meshgrid(x, y);
  H = meshwidth_graded(X, Y, meshwidth);

  ## surface plot
  #fig = plt.figure()
  #ax = fig.gca(projection='3d')
  #surf = ax.plot_surface(X, Y, H, rstride=1, cstride=1, antialiased=True)
  ##Show the plot
  #plt.show()
  
  X = np.reshape(X.transpose(), N*N) 
  Y = np.reshape(Y.transpose(), N*N) 
  #X = reshape(X, N*N, 1);
  #Y = reshape(Y, N*N, 1);

  # open file for appending
  fid = open(''.join(filename),'w')
  fid.write('View "background mesh" {\n')

  for i in range(N-1): 
    for j in range(N-1): 
      l = ['ST(' + str(x[i]) + ', ' + str(y[j]) + ', ' + '0, ']; 
      l.append(str(x[i+1]) + ', ' + str(y[j]) + ', 0, ');
      l.append(str(x[i+1]) + ', ' + str(y[j+1]) + ', 0)');
      l.append('{' + str(H[j][i]) + ', ' + str(H[j][i+1]) + ', ' + str(H[j+1][i+1]) + '};');
      s = ''.join(l) + '\n'
      fid.write(s);
 
  fid.write('};');
  fid.close();


def meshwidth_graded(x, y, h):

  beta = 4.0;
  R    = 0.75;

  # number of layers
  n = 1.0/(1.0 - (1.0 - h/R)**(1/beta));
  # smallest cell
  hmin = (1.0/n)**beta;
  
  r0 = np.sqrt((x+1.0)**2 + y**2);
  r1 = np.sqrt((x-1.0)**2 + y**2);

  I0 = np.nonzero(r0 < R);
  I1 = np.nonzero(r1 < R);
  Im = (np.logical_and(r0 >= R,r1 >= R)).nonzero();

  m = np.zeros(x.shape);

  for i in Im[0]:
    for j in Im[1]:
      m[i][j] = h;
  m[I0] = h*(r0[I0]/R)**(1.0-1.0/beta);
  m[I1] = h*(r1[I1]/R)**(1.0-1.0/beta);

  I = np.nonzero(m < hmin);
  m[I] = hmin;

  return m;


def main():
  
  h = 0.7;
  for i in range(9):
    createbgMesh(['bgmesh' + str(i) + '.pos'], [-1.0,1.0], [-1.0,1.0], h, 50);
    h = h/np.sqrt(2.0);
    #createbgMesh('bgmesh1.pos', [-1.0,1.0], [-1.0,1.0], 0.7);
    
    #x = np.linspace(-1.0, 1.0, 50);
    #y = np.linspace(-1.0, 1.0, 50);

    #[X,Y] = np.meshgrid(x, y);
    
    #m = meshwidth_graded(X, Y, 0.7);
    #print m.min()
    
if __name__ == '__main__':
    main()
    