import numpy as np
from numpy import linalg as LA

def data_generate(r,min,max):
	col_size = 5
	data = np.zeros(shape=[r,col_size])
	for index in range(r):
		mass = np.random.uniform(min,max,size=1)
		density = np.random.uniform(min,max,size=1)
		volume = mass / density
		cel = np.random.uniform(min,max,size=1)
		cel_f = cel*(9/5)+32 
		col = np.hstack([mass,density,volume,cel,cel_f])
		data[index] = col
	return data

def svd(A):
	At = np.transpose(A)	
	ata = np.dot(At,A)
	S,V = LA.eigh(ata)
	V = np.fliplr(V)
	V = np.transpose(V)
	S = S[(S >= 0 )]
	S = S[::-1]
	S = np.sqrt(np.diag(S))
	if(S.shape[1] != V.shape[0]):
		V = V[:S.shape[1]:]
	U = np.dot(np.dot(A,np.transpose(V)),LA.pinv(S))
	return U,S,V 

A = data_generate(100,1,100)

U,S,V = svd(A)

print "\n\nU=\n",U,"\n\nV=\n",V,"\n\nS=\n",S

for index in range(3):
	U = np.delete(U,U.shape[1]-1,axis=1)
	V = np.delete(V,V.shape[0]-1,axis=0)
	S = np.delete(S,S.shape[1]-1,axis=0)
	S = np.delete(S,S.shape[1]-1,axis=1)

reconstruction = np.dot(np.dot(U,S),V)
print "\n\nU=\n",U,"\n\nV=\n",V,"\n\nS=\n",S,"\n\nA=\n",A,"\n\nU*S*V=\n",reconstruction
