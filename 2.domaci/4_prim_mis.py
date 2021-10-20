import numpy as np
from numpy import linalg as la
import cv2 
from PIL import Image
from scipy.linalg import svd


coords = []


def afine(t):
    return [t[0]/t[2], t[1]/t[2]]

def homogene(t):
    return [t[0], t[1], 1]

def print_matrix(matrix, msg):

    print(msg)
    for line in matrix:
        print(round(line[0], 5), " ", round(line[1], 5), " ", round(line[2], 5), " ")

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


def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        coords.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ', ' + str(y), (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)   
        
        
#----------------------------------------------------------------------------------------
   

img = cv2.imread("/home/boris/FAXXX/4I/PPGR/2.domaci/left.png", cv2.IMREAD_COLOR)
cv2.imshow('image', img)
print(img.shape)
img = cv2.resize(img,(400,500))
cv2.imshow('image', img)
cv2.setMouseCallback('image', on_click)
while True:
    cv2.imshow("image",img)
    if cv2.waitKey(1) & len(coords) == 6:
        break

img = cv2.imread("/home/boris/FAXXX/4I/PPGR/2.domaci/right.png", 1)
img = cv2.resize(img,dsize=(400,500))
cv2.imshow('image', img)
cv2.setMouseCallback('image', on_click)
while True:
    cv2.imshow("image",img)
    if cv2.waitKey(1) & len(coords) == 12:
        break
cv2.destroyAllWindows()


points = []
for i in range(6):
    points.append(homogene(coords[i]))

prim_points = []
for i in range(6):
    coords[i+6][0] +=400
    prim_points.append(homogene(coords[i+6]))

zip_points = list(zip(points, prim_points))
PDLT = DLT_algorithm(zip_points)

img = cv2.imread("/home/boris/FAXXX/4I/PPGR/2.domaci/left.png", 1)
img = cv2.resize(img,dsize=(400,500))

img2 = cv2.imread("/home/boris/FAXXX/4I/PPGR/2.domaci/right.png", 1)
img2 = cv2.resize(img2,dsize=(400,500))

tmp_img = cv2.imread("/home/boris/FAXXX/4I/PPGR/2.domaci/right.png", 1)
tmp_img = cv2.resize(tmp_img,dsize=(400,500))

for i in range(400):
    for j in range(500):          
        tmp_img[j][i] = [0,0,0]
        
im = cv2.hconcat([tmp_img,img2])    
dodatak = cv2.hconcat([img,tmp_img])
PDLT = np.float32(PDLT)
sl = cv2.warpPerspective(dodatak,PDLT,(600,900))
   
for i in range(400):
    for j in range(500): 
        if(im[j,i,0] == 0 and im[j,i,1] == 0 and im[j,i,2] == 0):      
            im[j][i] = sl[j][i]

cv2.imshow('PANORAMA',im)
cv2.waitKey(0)       

imgs = [img,img2]
s = cv2.Stitcher.create()
res = s.stitch(imgs)
