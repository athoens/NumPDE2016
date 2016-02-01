import numpy as np
from numpy import sqrt
import matplotlib.pylab as plt
import meshes as msh
import FEM

def main():
  
  lu = 1.548888;                               # value of l(sol)
  f   = lambda x, y: (1.0);

  Nspacings = 9;                  # number of mesh widths to test (in powers of sqrt(2))           -> CHANGE BACK TO 10!!!!
  h0 = 0.3;                       # maximum value of mesh width
  h = [None]*Nspacings;           # vector for mesh widths (set values)
  dof = np.zeros(Nspacings);      # vector for mesh widths (true values)
  dof_reg = np.zeros(Nspacings);      # vector for mesh widths (true values)
  discret_err_bgm = [None]*Nspacings;     # vector for discretization errors for linear FEM
  discret_err_reg = [None]*Nspacings;    # vector for discretization errors for quadratic FEM

  # regular mesh
  for j in range(Nspacings):
    
    h = h0*2.0**(-j/2.0);
    [p, t, be, bd]=msh.create_msh("regular", h)
    
    # LINEAR FINITE ELEMENTS
    # Compute the discretized system:
    A_lin = FEM.stiffness(p, t);          # stiffness-matrix
    F_lin = FEM.load(p, t, 3, f);         # load-vector
    
    # degrees of freedom
    dof_reg[j] = p.shape[0]


    # solve system with Dirichtlet boundary conditions
    u_n = FEM.solve_d0(p, t, bd, A_lin, F_lin)

    # LINEAR FINITE ELEMENTS
    lun_lin = np.dot(u_n,F_lin);      # value of l(u_n)
    en_lin = lu-sqrt(abs(lun_lin));          # energy norm of error vector
    discret_err_reg[j] = en_lin;

  print 
  print "energy error with the regular mesh"
  print discret_err_reg

  # background mesh 
  for j in range(Nspacings):

    #[p, t, be, bd]=msh.create_msh_bgm("bgmesh"+str(j))
    [p, t, be, bd]=msh.read_gmsh("bgmesh"+str(j)+".msh")
    
    # LINEAR FINITE ELEMENTS
    # Compute the discretized system:
    A_lin = FEM.stiffness(p, t);          # stiffness-matrix
    F_lin = FEM.load(p, t, 3, f);         # load-vector
    
    # degrees of freedom
    dof[j] = p.shape[0]

    # solve system with Dirichtlet boundary conditions
    u_n = FEM.solve_d0(p, t, bd, A_lin, F_lin)

    # LINEAR FINITE ELEMENTS
    lun_lin = np.dot(u_n,F_lin);      # value of l(u_n)
    en_lin = lu-sqrt(abs(lun_lin));          # energy norm of error vector
    discret_err_bgm[j] = en_lin;

  print 
  print "energy error with the background mesh"
  print discret_err_bgm

  # ploting energy error distribution
  plt.loglog(dof_reg, discret_err_reg,':*', label='energy error - regular mesh')
  plt.loglog(dof, discret_err_bgm,':*', label='energy error - bgm')
  plt.title('energy error depending on degrees of freedom')
  plt.legend(loc=1)
  plt.show()

  # Visualization of the solution
  FEM.plot(p, t, u_n);             # approximated solution
  plt.show()


if __name__ == '__main__':
    main()
 
