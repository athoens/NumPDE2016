import scipy.sparse as sp
import numpy as np
#  
# gaussTriangle(n)
# 
# returns abscissas and weights for "Gauss integration" in the triangle with 
# vertices (-1,-1), (1,-1), (-1,1)
#
# input:
# n - order of the numerical integration (1 <= n <= 5)
#
# output:
# x - 1xp-array of abscissas, that are 1x2-arrays (p denotes the number of 
#     abscissas/weights)
# w - 1xp-array of weights (p denotes the number of abscissas/weights)
#
def gaussTriangle(n):

  if n == 1:
      x = [[-1/3., -1/3.]];
      w = [2.];
  elif n == 2:
      x = [[-2/3., -2/3.],
           [-2/3.,  1/3.],
           [ 1/3., -2/3.]];
      w = [2/3.,
           2/3.,
           2/3.];
  elif n == 3:
      x = [[-1/3., -1/3.],
           [-0.6, -0.6],
           [-0.6,  0.2],
           [ 0.2, -0.6]];
      w = [-1.125,
            1.041666666666667,
            1.041666666666667,
            1.041666666666667];
  elif n == 4:
      x = [[-0.108103018168070, -0.108103018168070],
           [-0.108103018168070, -0.783793963663860],
           [-0.783793963663860, -0.108103018168070],
           [-0.816847572980458, -0.816847572980458],
           [-0.816847572980458,  0.633695145960918],
           [ 0.633695145960918, -0.816847572980458]];
      w = [0.446763179356022,
           0.446763179356022,
           0.446763179356022,
           0.219903487310644,
           0.219903487310644,
           0.219903487310644];
  elif n == 5:
      x = [[-0.333333333333333, -0.333333333333333],
           [-0.059715871789770, -0.059715871789770],
           [-0.059715871789770, -0.880568256420460],
           [-0.880568256420460, -0.059715871789770],
           [-0.797426985353088, -0.797426985353088],
           [-0.797426985353088,  0.594853970706174],
           [ 0.594853970706174, -0.797426985353088]];
      w = [0.450000000000000,
           0.264788305577012,
           0.264788305577012,
           0.264788305577012,
           0.251878361089654,
           0.251878361089654,
           0.251878361089654];
  else:
      print ('numerical integration of order' + str(n) + 'not available');
      
  return x, w


#
# plot(p,t,u)
#
# plots the linear FE function u on the triangulation t with nodes p
#
# input:
# p  - Nx2 matrix with coordinates of the nodes
# t  - Mx3 matrix with indices of nodes of the triangles
# u  - Nx1 coefficient vector
#
def plot(p,t,u):
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_trisurf(p[:, 0], p[:, 1], t, u, cmap=plt.cm.jet)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('u')
  plt.show()



#Seite 74
def elemStiffness(p):
    DK=sp.lil_matrix((2,3))
    AK=sp.lil_matrix((3,3)) 
    det=abs((p[1,0]-p[0,0])*(p[2,1]-p[0,1])-(p[1,1]-p[0,1])*(p[2,0]-p[0,0]))
    DK[0,0]=(p[1,1]-p[2,1])
    DK[0,1]=(p[2,1]-p[0,1])
    DK[0,2]=(p[0,1]-p[1,1])
    DK[1,0]=(p[2,0]-p[1,0])
    DK[1,1]=(p[0,0]-p[2,0])
    DK[1,2]=(p[1,0]-p[0,0])
    K=det/2
    AK=1/(4*K)*DK.transpose()*DK
    return AK

def elemStiffnessP2(p):
    AK=sp.lil_matrix((6,6)) 
    det=abs((p[1,0]-p[0,0])*(p[2,1]-p[0,1])-(p[1,1]-p[0,1])*(p[2,0]-p[0,0]))
    K=det/2
    DK=sp.lil_matrix((2,3))
    GK=sp.lil_matrix((3,3)) 
    DK[0,0]=(p[1,1]-p[2,1])
    DK[0,1]=(p[2,1]-p[0,1])
    DK[0,2]=(p[0,1]-p[1,1])
    DK[1,0]=(p[2,0]-p[1,0])
    DK[1,1]=(p[0,0]-p[2,0])
    DK[1,2]=(p[1,0]-p[0,0])
    GK=1/(4*K**2)*DK.transpose()*DK #upper left quarter of new stiffness matrix remains almost the same except for adding K^2
    grad1=1/(2*K)*np.array([(p[2,1]-p[0,1]),(p[0,0]-p[2,0])])
    grad2=1/(2*K)*np.array([(p[0,1]-p[1,1]),(p[1,0]-p[0,0])])
    grad3=1/(2*K)*np.array([(p[1,1]-p[2,1]),(p[2,0]-p[1,0])]) #gradients of the bubblefunctions
    M1=sp.lil_matrix([[1/6*np.inner(grad1,(grad2+grad3)),1/6*np.inner(grad1,(grad3+grad1)),1/6*np.inner(grad1,(grad1+grad2))],
        [1/6*np.inner(grad2,(grad2+grad3)),1/6*np.inner(grad2,(grad1+grad3)),1/6*np.inner(grad2,(grad1+grad2))],
        [1/6*np.inner(grad3,(grad2+grad3)),1/6*np.inner(grad3,(grad1+grad3)),1/6*np.inner(grad3,(grad1+grad2))]]) #manual calculation of upper right and lower left quarters (symmetric)
    M2=sp.lil_matrix([[1/12*np.inner(grad1+grad3,grad1+grad3),1/12*(np.inner(grad1,grad2)+1/2*np.inner(grad3,grad3)),1/12*(np.inner(grad1,grad3)+1/2*np.inner(grad2,grad2))],
        [1/12*((np.inner(grad1,grad2))+1/2*np.inner(grad3,grad3)),1/12*np.inner(grad1+grad3,grad1+grad3),1/12*(np.inner(grad2,grad3)+1/2*np.inner(grad1,grad1))],
        [1/12*(np.inner(grad1,grad3)+1/2*np.inner(grad2,grad2)),1/12*(np.inner(grad2,grad3)+1/2*np.inner(grad1,grad1)),1/12*np.inner(grad1+grad2,grad1+grad2)]])
    #calculation of the lower right quarter 
    Q1= sp.hstack([GK,M1])
    Q2= sp.hstack([M1,M2])
    AK=sp.vstack([Q1,Q2]) #assembling the complete stiffnessmatrix
    return AK

def elemMass(p):
    det=(p[1,0]-p[0,0])*(p[2,1]-p[0,1])-(p[1,1]-p[0,1])*(p[2,0]-p[0,0])
    K=det/2
    MK=sp.lil_matrix((3,3))
    for i in range(0,3):
        for j in range(0,3):
            if i==j:
                MK[i,j]=K/6
            else:
                MK[i,j]=K/12
    return MK

def elemMassP2(p):
    det=(p[1,0]-p[0,0])*(p[2,1]-p[0,1])-(p[1,1]-p[0,1])*(p[2,0]-p[0,0])
    K=det/2
    M1=elemMass(p) # first 3x3 remains the same
    M2=sp.lil_matrix((3,3)) #upper rigth and lower left are identical
    for i in range (0,3):
        for j in range(0,3):
            if i==j:
                M2[i,j]=K/60
            else:
                M2[i,j]=K/30
    M3=sp.lil_matrix((3,3)) #lower right
    for i in range (0,3):
        for j in range(0,3):
            if i==j:
                M3[i,j]=K/90
            else:
                M3[i,j]=K/180
    MK=sp.lil_matrix((6,6))
    Q1= sp.hstack([M1,M2])  #assembling
    Q2= sp.hstack([M2,M3])
    MK=sp.vstack([Q1,Q2])
    return MK

def elemLoad(p,n,f):
    fK=sp.lil_matrix((3,1))
    z,w=gaussTriangle(n)
    det=abs((p[1,0]-p[0,0])*(p[2,1]-p[0,1])-(p[1,1]-p[0,1])*(p[2,0]-p[0,0]))
    for i in range(0,3):
        sum=0
        for j in range(0,len(z)):
            k=z[j]
            xi1=(k[0]+1)/2
            xi2=(k[1]+1)/2
            N=np.array([1-xi1-xi2,xi1,xi2])
            x1=p[0,0]+xi1*(p[1,0]-p[0,0])+xi2*(p[2,0]-p[0,0])
            x2=p[0,1]+xi1*(p[1,1]-p[0,1])+xi2*(p[2,1]-p[0,1])
            sum=sum+w[j]*f(x1,x2)*N[i]
        fK[i]=sum*det/4
    return fK

def elemLoadP2(p,n,f): #remains the same except for 3 new shapefunctions
    fK=sp.lil_matrix((6,1))
    z,w=gaussTriangle(n)
    det=abs((p[1,0]-p[0,0])*(p[2,1]-p[0,1])-(p[1,1]-p[0,1])*(p[2,0]-p[0,0]))
    for i in range(0,6):
        sum=0
        for j in range(0,len(z)):
            k=z[j]
            xi1=(k[0]+1)/2
            xi2=(k[1]+1)/2
            N=np.array([1-xi1-xi2,xi1,xi2,xi1*xi2,(1-xi1-xi2)*xi2,(1-xi1-xi2)*xi1])  # 6 shapefunctions
            x1=p[0,0]+xi1*(p[1,0]-p[0,0])+xi2*(p[2,0]-p[0,0])
            x2=p[0,1]+xi1*(p[1,1]-p[0,1])+xi2*(p[2,1]-p[0,1])
            sum=sum+w[j]*f(x1,x2)*N[i]
        fK[i]=sum*det/4
    return fK

def stiffness(p,t):
    AK=sp.lil_matrix((len(p),len(p)))
    v=sp.lil_matrix((3,2))
    for i in range(0,len(t)):
        Tk=sp.lil_matrix((3,len(p)))
        for j in range(0,3):
            v[j,0]=p[t[i,j],0]
            v[j,1]=p[t[i,j],1]
            Tk[j,t[i,j]]=1      
        AK=AK+Tk.transpose()*elemStiffness(v)*Tk
    return AK

def stiffnessP2(p,t,e,l):
    AK=sp.lil_matrix((len(p)+l,len(p)+l))
    v=sp.lil_matrix((3,2))
    for i in range(0,len(t)):
        Tk=sp.lil_matrix((6,len(p)+l))
        for j in range(0,3):
            v[j,0]=p[t[i,j],0]
            v[j,1]=p[t[i,j],1]
            if e[t[i,j],t[i,(j+1)%3]]!=0:   #checking if we need to use edge (x,y) or (y,x) (only one has been assigned a number)
                g=e[t[i,j],t[i,(j+1)%3]]
            else:
                g=e[t[i,(j+1)%3],t[i,j]]
            Tk[j,t[i,j]]=1
            Tk[j+3,g+len(p)-1]=1    #adding 3 further rows for the edges
        AK=AK+Tk.transpose()*elemStiffnessP2(v)*Tk
    return AK

def mass(p,t):
    AK=sp.lil_matrix((len(p),len(p)))
    v=sp.lil_matrix((3,2))
    for i in range(0,len(t)):
        Tk=sp.lil_matrix((3,len(p)))
        for j in range(0,3):
            v[j,0]=p[t[i,j],0]
            v[j,1]=p[t[i,j],1]
            Tk[j,t[i,j]]=1
        AK=AK+Tk.transpose()*elemMass(v)*Tk
    return AK
def massP2(p,t,e,l):   #see stiffness p2
    AK=sp.lil_matrix((len(p)+l,len(p)+l))
    v=sp.lil_matrix((3,2))
    for i in range(0,len(t)):
        Tk=sp.lil_matrix((6,len(p)+l))
        for j in range(0,3):
            v[j,0]=p[t[i,j],0]
            v[j,1]=p[t[i,j],1]
            if e[t[i,j],t[i,(j+1)%3]]!=0:
                g=e[t[i,j],t[i,(j+1)%3]]
            else:
                g=e[t[i,(j+1)%3],t[i,j]]
            Tk[j,t[i,j]]=1
            Tk[j+3,g+len(p)-1]=1
        AK=AK+Tk.transpose()*elemMassP2(v)*Tk
    return AK


def load(p,t,n,f):
    AK=sp.lil_matrix((len(p),1))
    v=sp.lil_matrix((3,2))
    for i in range(0,len(t)):
        Tk=sp.lil_matrix((3,len(p)))
        for j in range(0,3):
            v[j,0]=p[t[i,j],0]
            v[j,1]=p[t[i,j],1]
            Tk[j,t[i,j]]=1
        AK=AK+Tk.transpose()*elemLoad(v,n,f)      
    return AK

def loadP2(p,t,n,f,e,l): #see stiffnesp2
    AK=sp.lil_matrix((len(p)+l,1))
    v=sp.lil_matrix((3,2))
    for i in range(0,len(t)):
        Tk=sp.lil_matrix((6,len(p)+l))
        for j in range(0,3):
            v[j,0]=p[t[i,j],0]
            v[j,1]=p[t[i,j],1]
            if e[t[i,j],t[i,(j+1)%3]]!=0:
                g=e[t[i,j],t[i,(j+1)%3]]
            else:
                g=e[t[i,(j+1)%3],t[i,j]]
            Tk[j,t[i,j]]=1
            Tk[j+3,g+len(p)-1]=1    
        AK=AK+Tk.transpose()*elemLoadP2(v,n,f)
    return AK

def interiorNodes(p,t,be):
    IN=np.zeros(len(p)-len(be))
    j=0
    for i in range(0,len(p)-len(be)):
            while j in be:
                j=j+1
            IN[i]=j
            j=j+1
    return IN


def elemLoadNeumann(p,n,g):
    gK=sp.lil_matrix((2,1))
    z,w=np.polynomial.legendre.leggauss(n) #z sample w weight
    det=np.sqrt((p[1,0]-p[0,0])**2+(p[1,1]-p[0,1])**2)
    for i in range(0,2):
        sum=0
        for j in range(0,len(z)):
            k=z[j]
            xi1=(k+1)/2
            N=np.array([1-xi1,xi1])
            x1=p[0,0]+xi1*(p[1,0]-p[0,0])
            x2=p[0,1]+xi1*(p[1,1]-p[0,1])
            sum=sum+w[j]*g(x1,x2)*N[i]
        gK[i]=sum*det/4
    
    return gK    
    
def loadNeumann(p,be,n,g):
    AK=sp.lil_matrix((len(p),1))
    v=sp.lil_matrix((2,2))
    for i in range(0,len(be)):
        Tk=sp.lil_matrix((2,len(p)))
        for j in range(0,2):
            v[j,0]=p[be[i,j]-1,0]
            v[j,1]=p[be[i,j]-1,1]
            Tk[j,be[i,j]-1]=1
        AK=AK+Tk.transpose()*elemLoadNeumann(v,n,g)
    return AK

    

        


