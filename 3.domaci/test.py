import sys
import numpy as np
from math import cos, acos, sin, asin, tan, atan, atan2, pi, sqrt

def print_matrix(matrix, msg):
    print(msg)
    for line in matrix:
        print(round(line[0], 5), " ", round(line[1], 5), " ", round(line[2], 5), " ")
    
    print()
    
def Euler2A(fi, teta, psi):
    Rx = np.array([[1, 0, 0],[0, cos(fi), -sin(fi)],[0, sin(fi), cos(fi)]])
    Ry = np.array([[cos(teta), 0, sin(teta)], [0, 1, 0], [-sin(teta), 0, cos(teta)]])
    Rz = np.array([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
    
    return np.dot(Rz, np.dot(Ry, Rx))

def AxisAngle(A):
    E = np.identity(3)
    AxAT = np.dot(A, np.transpose(A))
    AxAT = np.round(AxAT, 1)
    
    if not np.all(np.equal(E, AxAT)):
        sys.exit('Matrica A nije ortogonalna')
    if not np.round(np.linalg.det(A),1) == 1:
        sys.exit('Matrica A nije determinantna nije 1')
        
    A_copy = A
    A = A - E

    p = np.cross(A[0], A[1])
    if np.array_equal(p, np.zeros(3)):
        p = np.cross(A[0], A[2])
        if np.array_equal(p, np.zeros(3)):
            p = np.cross(A[1], A[2])
            if np.array_equal(p, np.zeros(3)):
                return None, None

    p = p / np.linalg.norm(p)
    x = np.array([-p[1], p[0], 0])
    x = x / np.linalg.norm(x)
    x_prim = np.dot(A_copy, x)
    fi = acos(np.dot(x, x_prim))
    
    det = np.linalg.det([x, x_prim, p])
    if det < 0:
        p = -p
    
    return p, fi
    
def Rodrigez(p, fi):
    sum1 = np.outer(p, p.T)
    
    E = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    sum2 = cos(fi) * (E - sum1)
    
    px =  np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])
    sum3 = sin(fi) * px
    
    return sum1 + sum2 + sum3
    
def A2Euler(A):
    E = np.identity(3)
    AxAT = np.dot(A, np.transpose(A))
    AxAT = np.round(AxAT, 1)
    
    if not np.all(np.equal(E, AxAT)):
        sys.exit('Matrica A nije ortogonalna')
    if not np.round(np.linalg.det(A),1) == 1:
        sys.exit('Matrica A nije determinantna nije 1')
        
    if A[2][0] < 1:
        if A[2][0] > -1:
            psi = atan2(A[1][0], A[0][0])
            teta = asin(-A[2][0])
            fi = atan2(A[2][1], A[2][2])
        else:
            psi = atan2(-A[0][1], A[1][1])
            teta = pi/2
            fi = 0
    else:
        psi = atan2(-A[0][1], A[1][1])
        teta = -pi/2
        fi = 0
    
    return psi, teta, fi
    
def AxisAngle2Q(p, fi):
    l = fi/2
    return (sin(l)*p, cos(l))

def Q2AxisAngle(q):
    #q = ([i, j, k], w)
    v = q[0]
    w = q[1]
    i = v[0] 
    j = v[1] 
    k = v[2]
    norm = sqrt(i**2 + j**2 + k**2 + w**2)
    v = v / norm
    w = w / norm
    
    l = acos(w)
    p = v / sin(l)
    fi = 2*l
    return p, fi
    
###############################################################################################

print('Ulazni uglovi:')
fi = -atan(1/4)
teta = -asin(8/9)
psi = atan(4)
print('psi: ', psi, 'teta: ', teta, 'fi: ', fi)

print()

A = Euler2A(fi, teta, psi)
print_matrix(A, 'Euler2A')

print('AxisAngle')
p, fi = AxisAngle(A)
print('p: ', p, 'fi :', fi)

print()

A = Rodrigez(p, fi)
print_matrix(A, 'Rodrigez A')

print('A2Euler')
psi, teta, fi = A2Euler(A)
print('psi: ', psi, 'teta: ', teta, 'fi: ', fi)

print()

print('AxisAngle2Q')
fi = pi/2
q = AxisAngle2Q(p, fi)
print('q: ', q)

print()

print('Q2AxisAngle')
p, fi = Q2AxisAngle(q)
print('p: ', p, 'fi: ', fi)
