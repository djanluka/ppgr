import numpy as np
import numpy.linalg as la

n=7 #161/2017
t = np.array([[5, -1-2*n, 3, 18-3*n], [0,-1, 5, 21], [0, -1, 0, 1]])
print("T\n", t)
c1 = t[:, 1:]
c2 = -t[:, [0,2,3]]
c3 = t[:, [0,1,3]]
c4 = -t[:, [0,1,2]]

c1 = la.det(c1)
c2 = la.det(c2)
c3 = la.det(c3)
c4 = la.det(c4)

c = np.array([c1/c4, c2/c4, c3/c4])
print('Kamera ', c)
to = t[:, [0,1,2]]
q, r = la.qr(la.inv(to))

if r[0][0] < 0:
    r[0][0] = -r[0][0]
    r[0][1] = -r[0][1]
    r[0][2] = -r[0][2]
    q[0][0] = -q[0][0]
    q[1][0] = -q[1][0]
    q[2][0] = -q[2][0]
elif r[1][1] < 0:
    r[1][1] = -r[1][1]
    r[1][2] = -r[1][2]
    q[0][1] = -q[0][1]
    q[1][1] = -q[1][1]
    q[2][1] = -q[2][1]
elif r[2][2] < 0:
    r[2][2] = -r[2][2]
    q[0][2] = -q[0][2]
    q[1][2] = -q[1][2]
    q[2][2] = -q[2][2]

k = la.inv(r)
a = la.inv(q)
print('K\n', k)
print('A\n', a)

def print_matrix(matrix, msg):

    print(msg)
    for line in matrix:
        print(round(line[0], 5), " ", round(line[1], 5), " ", round(line[2], 5), " ")

def create_two_eq(x, xp):
    first_eq = [0, 0, 0, 0]
    for xs in x:
        first_eq.append(-xp[2] * xs)
    for xs in x:
        first_eq.append(xp[1] * xs)
    #[x3'*x1, x3'*x2, x3'*x3, 0, 0, 0, -x1'*x1, -x1'*x2, -x1'*x3]    
    second_eq = []
    for xs in x:
        second_eq.append(xp[2] * xs)
    second_eq.append(0)
    second_eq.append(0)
    second_eq.append(0)
    second_eq.append(0)
    for xs in x:
        second_eq.append(-xp[0] * xs)
    
    return first_eq, second_eq
    
def DLT_algorithm(zip_points):
    #Odredjujemo matricu A
    #Za svaku tacku i njenu sliku dodajemo dve jednacine
    matrix_A = []
    for points in zip_points:
        x = points[0]
        xp = points[1]
        first_eq, second_eq = create_two_eq(x, xp)
        matrix_A.append(first_eq)
        matrix_A.append(second_eq)
    
    _, _, VT = la.svd(matrix_A)
    #print(matrix_A)
    #Resenje je poslednja vrsta iz VT matrice
    t = VT[-1]
    t = t/t[0]
    print("T preko svd\n", t)

m1 = np.array([460,280,250,1])
m2 = np.array([50,380,350,1])
m3 = np.array([470,500,100,1])
m4 = np.array([380,630,50*n,1])
m5 = np.array([30*n,290,0,1])
m6 = np.array([580,0, 130, 1])

m1p = np.array([288,251,1])
m2p = np.array([79,510,1])
m3p = np.array([470,440,1])
m4p = np.array([520,590,1])
m5p = np.array([365,388,1])
m6p = np.array([365,20, 1])

points = [m1,m2,m3,m4,m5,m6]
prim_points = [m1p,m2p,m3p,m4p,m5p,m6p]
zip_points = list(zip(points, prim_points))
DLT_algorithm(zip_points)
