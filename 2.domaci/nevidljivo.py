import numpy as np

def afine(p):
    return np.array([round(p[0]/p[2]), round(p[1]/p[2]), 1])

def nevidljivo(p1, p2, p3, p5, p6, p7, p8):
    #pravimo homogene koordinate
    p1 = np.append(p1, 1)
    p2 = np.append(p2, 1)
    p3 = np.append(p3, 1)
    p5 = np.append(p5, 1)
    p6 = np.append(p6, 1)
    p7 = np.append(p7, 1)
    p8 = np.append(p8, 1)
    
    prava_2_6 = np.cross(p2, p6)
    prava_1_5 = np.cross(p1, p5)
    prava_5_6 = np.cross(p5, p6)
    prava_7_8 = np.cross(p7, p8)
    
    xb = np.cross(prava_2_6, prava_1_5)
    yb = np.cross(prava_5_6, prava_7_8)
    
    prava_xb_8 = np.cross(xb, p8)
    prava_yb_3 = np.cross(yb, p3)
    
    return np.cross(prava_xb_8, prava_yb_3)
    
p1 = np.array([204, 160])
p2 = np.array([91, 261])
p3 = np.array([38, 198])
p5 = np.array([235, 101])
p6 = np.array([83, 226])
p7 = np.array([15, 147])
p8 = np.array([168, 56])

p4 = afine(nevidljivo(p1, p2, p3, p5, p6, p7, p8))
#[156 114 1]
print(p4)

