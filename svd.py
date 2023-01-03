import numpy as np

mat1=np.array([[12,23,22],[5,87,34],[44,77,3]])
mat2=np.array([[12,32,22],[7,78,43],[44,77,3]])

print('Addition')
print(np.add(mat1,mat2))
print(np.subtract(mat1,mat2))
print(np.divide(mat1,mat2))
print(np.multiply(mat1,mat2))

from scipy.linalg import svd
A = np.array([[12, 21, 39], [94, 75, 46], [37, 80, 94], [64, 34, 99], [38, 12, 89]])
U,s,VT = svd(A)
print("Decomposed matrix:\n",U)
print("Inverse matrix\n",s)
print('Transponse matrix:\n',VT)