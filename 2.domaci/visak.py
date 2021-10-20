
#INVARIJANTNOST U ODNOSU NA PROMENU KOORDINATA
'''
a = np.array([-3, 2, 1])
b = np.array([-2, 5, 2])
c = np.array([1, 0, 3])
d = np.array([-7, 3, 1])

ap = np.array([11, -12, 7])
bp = np.array([25, -8, 9])
cp = np.array([15, 4, 17])
dp = np.array([14, -28, 10])

P1 = naive_algorithm(a, b, c, d, ap, bp, cp, dp)

e = np.array([2, 1, 2])
f = np.array([-1, 2, 1])
g = np.array([1, 1, 1])
ep = np.dot(P1, e)
fp = np.dot(P1, f)
gp = np.dot(P1, g)

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
M1s = np.dot(np.linalg.inv(cc), np.dot(M1, cc))
print_matrix(M1, "DLT:Matrica koja dokazuje imunost na invarijantnost:")

M1 = normalized_DLT_algorithm(points, prim_points)
M1s = np.dot(np.linalg.inv(cc), np.dot(M1, cc))
print_matrix(M1, "DLT_NORM: Matrica koja dokazuje imunost na invarijantnost:")
'''
