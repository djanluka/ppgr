#%%
import numpy as np
from scipy.linalg import svd
import math

# %%
def print_matrix(matrix, msg):

    print(msg)
    for line in matrix:
        print(round(line[0], 5), " ", round(line[1], 5), " ", round(line[2], 5), " ")


# %%
def naive_algorithm(a, b, c, d, ap, bp, cp, dp):
    P1 = get_matrix_P(a, b, c, d, ap, bp, cp, dp)
    print_matrix(P1, 'Matrica projektivnog preslikavanja 4 tacke:')
    return P1

# %%
def get_matrix_P(a, b, c, d, ap, bp, cp, dp):
    #Odredimo a,b,c != 0 tako da D=aA+bB+cC
    alpha_beta_gama = get_alpha_beta_gama(a,b,c,d)
    #P1 je matrica sa kolonama aA, bB, cC
    p1 = get_matrix_transformation(alpha_beta_gama, a, b, c)
    
    #Odredimo a',b',c' != 0 tako da D'=a'A'+b'B'+c'C'
    prim_alpha_beta_gama = get_alpha_beta_gama(ap, bp, cp, dp)
    #P1 je matrica sa kolonama a'A', b'B', c'C'
    p2 = get_matrix_transformation(prim_alpha_beta_gama, ap, bp, cp)
    
    p1_inv = np.linalg.inv(p1)
    #resenje je P=P2*P1^(-1)
    return np.dot(p2, p1_inv)

# %%
def get_alpha_beta_gama(a,b,c,d):
    A = np.array([a, b, c])
    A = np.transpose(A)
    B = d
    return np.linalg.solve(A, B)

# %%
def get_matrix_transformation(alpha_beta_gama, a, b, c):
    alpha = alpha_beta_gama[0]
    beta = alpha_beta_gama[1]
    gama = alpha_beta_gama[2]
    p = np.array([alpha*a, beta*b, gama*c])
    return np.transpose(p)
# %%   
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
    
    _, _, VT = svd(matrix_A)
    
    #Resenje je poslednja vrsta iz VT matrice
    P2 = np.reshape(VT[-1], (3,3))
    print_matrix(P2, 'Matrica DLT algoritma sa vise od 4 tacke:')
    return P2
# %%
def create_two_eq(x, xp):
    #[0,0,0 -x3'*x1, -x3'*x2, -x3'*x3, x2'*x1, x2'*x2, x2'*x3]
    first_eq = [0, 0, 0]
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
    for xs in x:
        second_eq.append(-xp[0] * xs)
    
    return first_eq, second_eq

# %%
def afine(p):
    return np.array([np.double(p[0]/p[2]), np.double(p[1]/p[2])])

# %%
def get_matrix_T(points):
    #Od polaznih homogenih tacaka, pravimo afine tacke
    afine_points = []
    for p in points:
        afine_points.append(afine(p))
    
    #C je teziste
    C = sum(afine_points) / len(afine_points)
    #G je matrica translacije za teziste 
    G = np.array([[1, 0, -C[0]], [0, 1, -C[1]], [0, 0, 1]])
    
    norm_afine_points = []
    for p in afine_points:
        vector_cp = p - C
        norm_afine_points.append(math.sqrt(vector_cp[0]**2 + vector_cp[1]**2))

    l = sum(norm_afine_points) / len(norm_afine_points)
    
    #S je matrica skalirana sa korenom iz dva
    S = np.array([[math.sqrt(2)/l, 0, 0], [0, math.sqrt(2)/l, 0], [0, 0, 1]])
    
    #T = S*G
    print_matrix(np.dot(S, G), 'Matrica koja normalizuje:')
    return np.dot(S, G)


# %%
def normalized_DLT_algorithm(points, prim_points):
    T = get_matrix_T(points)
    a_norm = np.dot(T, points[0])
    b_norm = np.dot(T, points[1])
    c_norm = np.dot(T, points[2])
    d_norm = np.dot(T, points[3])
    e_norm = np.dot(T, points[4])
    f_norm = np.dot(T, points[5])

    T_prim = get_matrix_T(prim_points)
    a_prim_norm = np.dot(T_prim, prim_points[0])
    b_prim_norm = np.dot(T_prim, prim_points[1])
    c_prim_norm = np.dot(T_prim, prim_points[2])
    d_prim_norm = np.dot(T_prim, prim_points[3])
    e_prim_norm = np.dot(T_prim, prim_points[4])
    f_prim_norm = np.dot(T_prim, prim_points[5])

    norm_points = [a_norm, b_norm, c_norm, d_norm, e_norm, f_norm]
    norm_prim_points = [a_prim_norm, b_prim_norm, c_prim_norm, d_prim_norm, e_prim_norm, f_prim_norm]
    zip_points = list(zip(norm_points, norm_prim_points))
    P3 = DLT_algorithm(zip_points)

    #Normalizovana DLT matrica P = T' * P3 * T^(-1)
    goal_matrix = np.dot(np.linalg.inv(T_prim), np.dot(P3,T))
    print_matrix(goal_matrix, 'Normalizovana DLT matrica preslikavanja:')
    return goal_matrix

# %%
def check_algorithm(dlt, p, coords, msg):
    #Provera matrica dobijenih DLT i NDLT algoritmom
    #Tako sto delimo dlt matricu sa nekim ne-nula elementom
    #Zatim mnozimo za elementom iz matrice P sa iste pozicije
    dlt = dlt / dlt[coords[0]][coords[1]]
    dlt = dlt * p[coords[0]][coords[1]]
    print_matrix(dlt, msg)


#PRVI TEST PRIMER
'''
a = np.array([-3, -1, 1])
b = np.array([3, -1, 1])
c = np.array([1, 1, 1])
d = np.array([-1, 1, 1])

ap = np.array([-2, -1, 1])
bp = np.array([2, -1, 1])
cp = np.array([2, 1, 1])
dp = np.array([-2, 1, 1])

P = naive_algorithm(a, b, c, d, ap, bp, cp, dp)

e = np.array([1, 2, 3])
f = np.array([-8, -2, 1])
ep = np.dot(P, e)
fp = np.dot(P, f)

points = [a,b,c,d,e,f]
prim_points = [ap,bp,cp,dp,ep,fp]
zip_points = list(zip(points, prim_points))

PDLT = DLT_algorithm(zip_points)
check_algorithm(PDLT, P, (1,1),"Provera za PDLT i P:")

NDLT = normalized_DLT_algorithm(points, prim_points)
check_algorithm(NDLT, P, (1,1), "Provera za NDLT i P:")
'''

#TRECI TEST PRIMER
'''
a = np.array([-3, 2, 1])
b = np.array([-2, 5, 2])
c = np.array([1, 0, 3])
d = np.array([-7, 3, 1])

ap = np.array([11, -12, 7])
bp = np.array([25, -8, 9])
cp = np.array([15, 4, 17])
dp = np.array([14, -28, 10])
P = naive_algorithm(a, b, c, d, ap, bp, cp, dp)

e = np.array([2, 1, 2])
f = np.array([-1, 2, 1])
g = np.array([1, 1, 1])
ep = np.dot(P, e)
fp = np.dot(P, f)
gp = np.dot(P, g)
gp[0] = 8.02

points = [a,b,c,d,e,f,g]
prim_points = [ap,bp,cp,dp,ep,fp,gp]
zip_points = list(zip(points, prim_points))
PDLT = DLT_algorithm(zip_points)
check_algorithm(PDLT, P, (0,1),"Provera za PDLT i P:")

NDLT = normalized_DLT_algorithm(points, prim_points)
check_algorithm(NDLT, P, (0,1),"Provera za NDLT i P:")
'''

#PRIMER ZA INVARIJANTNOST KOORDINATA
'''
a = np.array([-3, 2, 1])
b = np.array([-2, 5, 2])
c = np.array([1, 0, 3])
d = np.array([-7, 3, 1])

ap = np.array([11, -12, 7])
bp = np.array([25, -8, 9])
cp = np.array([15, 4, 17])
dp = np.array([14, -28, 10])
P = naive_algorithm(a, b, c, d, ap, bp, cp, dp)

e = np.array([2, 1, 2])
f = np.array([-1, 2, 1])
g = np.array([1, 1, 1])
ep = np.dot(P, e)
fp = np.dot(P, f)
gp = np.dot(P, g)
gp[0] = 8.02

points = [a,b,c,d,e,f,g]
prim_points = [ap,bp,cp,dp,ep,fp,gp]
zip_points = list(zip(points, prim_points))
PDLT = DLT_algorithm(zip_points)

M = PDLT / PDLT[0][1]
M = M * 3

NDLT = normalized_DLT_algorithm(points, prim_points)
NM = NDLT / NDLT[0][1]
NM = NM * 3

#vracamo pravu vrednost za gp
gp = np.dot(P, g)

#MATRICA INVARIJANTNOSTI
cc = np.array([[0,1,2], [-1,0,3], [0,0,1]])
a = np.dot(cc, a)
b = np.dot(cc, b)
c = np.dot(cc, c)
d = np.dot(cc, d)
e = np.dot(cc, e)
f = np.dot(cc, f)
g = np.dot(cc, g)
ap = np.dot(cc, ap)
bp = np.dot(cc, bp)
cp = np.dot(cc, cp)
dp = np.dot(cc, dp)
ep = np.dot(cc, ep)
fp = np.dot(cc, fp)
gp = np.dot(cc, gp)

points = [a,b,c,d,e,f,g]
prim_points = [ap,bp,cp,dp,ep,fp,gp]
zip_points = list(zip(points, prim_points))
M1 = DLT_algorithm(zip_points)

print()
print('-------------------------------------------------------')
M1s = np.dot(np.linalg.inv(cc), np.dot(M1, cc))
print_matrix(M1s, "M1s = Inverse[cc].m1.cc")
print()

MM = M1s / M1s[0][1]
MM = MM * 3
print_matrix(MM, "MM = (M1s / M1s[0][1]) * 3")
print()

print_matrix(M, 'M')
print()
print("Vidimo da matrice MM i M NISU sasvim iste sto govori da DLT NIJE invarijantan na promenu koordinata")
print('-------------------------------------------------------')


M2 = normalized_DLT_algorithm(points, prim_points)

print()
print('-------------------------------------------------------')
M2s = np.dot(np.linalg.inv(cc), np.dot(M2, cc))
print_matrix(M2s, "M2s = Inverse[cc].m2.cc")
print()

NM2 = M2s / M2s[0][1]
NM2 = NM2 * 3
print_matrix(NM2, "NM2 = (M2s / M2s[0][1]) * 3")
print()

print_matrix(NM, 'NM')
print()
print('Vidimo da su matrice NM2 i NM JESU sasvim iste sto govori da normalizovani DLT JESTE invarijantan na promenu koordinata')
print('-------------------------------------------------------')
'''
