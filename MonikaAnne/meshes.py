import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import scipy.sparse as sp

def read_gmsh(msh):
	m=open(msh)
	d=m.readlines()
	l=len(d)
	N=0 #Anzahl Knoten
	M=0 #Anzahl 3ecke
	G=0 #Anzahl Elemente
	B=0 #Anzahl Randlinien
	k=0 #Zeile, wo Knoten anfangen
	j=0 #Zeile, wo Elemente anfangen
	for i in range(0,l):
		if d[i]=='$Nodes\n':
			N=int(d[i+1])
			k=i+2
		if d[i]=='$Elements\n':
			G=int(d[i+1])
			j=i+2
	for i in range(j,j+G):
		if len(d[i].split())>1 and d[i].split()[1]=='1':
			B=B+1	
		if len(d[i].split())>1 and d[i].split()[1]=='2':
			M=M+1
	be=np.zeros((B,2))
	p=np.zeros((N,2))
	t=np.zeros((M,3))
	for i in range(0,N):	#trage Knotenkoordinaten ein
		l=d[k+i] 
		w=l.split() 		
		for n in range(0,2):
			p[i,n]=w[n+1]
	for i in range(0,M):	#trage 3eckskoordinaten ein
		l=d[j+G-M+i]
		w=l.split() 		
		for n in range(0,3):
			t[i,n]=int(w[-3+n])-1		
	for i in range(0,B):	#trage linienpunkte ein
		l=d[j+4+i] 
		w=l.split()
		#print(str(j)+"....j")
		#print(G)
		#print(M)
		#print(B)
		#print(i) 	
		#print(j+G-M-B+i)
		for n in range(0,2):
			be[i,n]=int(w[-2+n])
	m.close
	return p,t,be


def grid_square(a,h0):
	#h skalieren
	seite=(np.sqrt(2)*a)/h0
	h=a*np.sqrt(2)/np.ceil(seite) 
	#grössen festlegen
	s=h/np.sqrt(2) #Seitenlaenge eines kleinen Quadrats
	N=((a/s)+1)**2	#Anzahl Nodes
	M=2*(a**2)/(s**2)#Anzahl Triangles
	B=4*a/s
	be=np.zeros((B,2)) 
	p=np.zeros((N, 2))
	t=np.zeros((M,3))
	wN=int(np.sqrt(N))
	
#erzeuge punkte		
	for i in range(0,int(N)):
		p[i,0]=(i//wN)*s	
		p[i,1]=i*s-(i//wN)*wN*s
#erzeuge je 2 dreiecke pro punkt
	j=0
	for i in range(0,int(N)-(wN+1)):
		if (i+1)%wN!=0 or i==0: #schliesse randpunkte aus
			t[j,0]=i		#trage punktindizes ein
			t[j,1]=wN+i
			t[j,2]=wN+i+1 #untere dreiecke
			j=j+1
			t[j,0]=i
			t[j,1]=wN+i+1
			t[j,2]=i+1
			j=j+1
			
	for j in range(0,int(wN)-1): #links
		be[j,0]=j
		be[j,1]=j+1
	for j in range(0,int(wN)-1): #oben
		be[wN-1+j,0]=(j+1)*wN-1
		be[wN-1+j,1]=(j+2)*wN-1
	for j in range(0,int(wN)-1): #rechts
		be[2*(wN-1)+j,0]=(wN-1)*wN+j
		be[2*(wN-1)+j,1]=(wN-1)*wN+j+1
	for j in range(0,int(wN)-1): #unten
		be[3*(wN-1)+j,0]=j*wN
		be[3*(wN-1)+j,1]=(j+1)*wN
	
	return (p,t,be)
def show(p,t):
	M=len(t)
	N=len(p)
	fig = plt.figure()
	ax = fig.add_subplot(111)	
	#punkteplot
	for i in range(0,N):
		plt.plot(p[i,0],p[i,1],'ok')
	#3eckplot
	for i in range(0,M):	
		a=[p[t[i,0],0],p[t[i,0],1]]
		b=[p[t[i,1],0],p[t[i,1],1]]
		c=[p[t[i,2],0],p[t[i,2],1]]
		l1=lines.Line2D([a[0],b[0]],[a[1],b[1]],color='k') 
		l2=lines.Line2D([b[0],c[0]],[b[1],c[1]],color='k')
		l3=lines.Line2D([c[0],a[0]],[c[1],a[1]],color='k')
		ax.add_line(l1) 
		ax.add_line(l2) 
		ax.add_line(l3)
	plt.axes().set_aspect('equal', 'datalim')
	plt.show()

#g
def max_mesh_width(p,t):
	M=len(t)
	N=len(p)
	h1=0
	h2=0
	h3=0
	maxi=0
	maximum=0
	for i in range (0,M):
		x1=p[t[i,0],0] 
		x2=p[t[i,1],0]
		x3=p[t[i,2],0]
		y1=p[t[i,0],1]
		y2=p[t[i,1],1]
		y3=p[t[i,2],1]
		h1=np.sqrt((x3-x1)**2+(y3-y1)**2)
		h2=np.sqrt((x2-x1)**2+(y2-y1)**2)
		h3=np.sqrt((x3-x2)**2+(y3-y2)**2)
		maxi=max(h1,h2,h3)
		maximum=max(maxi,maximum)
	return maximum

#2
def edgeIndex(p,t):
	E=sp.lil_matrix((len(p),len(p)))  	
	j=1			#running index for the edges
	for i in range (0,len(t)):
		if E[t[i,0],t[i,1]]==0 and E[t[i,1],t[i,0]]==0: #filter if edge already has a number 
			E[t[i,0],t[i,1]]=j
			j=j+1
		if E[t[i,1],t[i,2]]==0 and E[t[i,2],t[i,1]]==0:
			E[t[i,1],t[i,2]]=j
			j=j+1
		if E[t[i,2],t[i,0]]==0 and E[t[i,0],t[i,2]]==0:
			E[t[i,2],t[i,0]]=j
			j=j+1
	return E,(j-1) #return matrix and number of edges