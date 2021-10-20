import numpy as np
from numpy import linalg as la
import cv2 
from PIL import Image


unete_koordinate = []


def u_afine(t):
    return [t[0]/t[2], t[1]/t[2]]

def u_homogene(t):
    return [t[0], t[1], 1]

def korespodencija(t, pt):
    Mi = np.matrix([
        [0, 0, 0, -pt[2] * t[0], -pt[2] * t[1], -pt[2] * t[2], pt[1] * t[0], pt[1] * t[1], pt[1] * t[2]],
        [pt[2] * t[0], pt[2] * t[1], pt[2] * t[2], 0, 0, 0, -pt[0] * t[0], -pt[0] * t[1], -pt[0] * t[2]]])
    return Mi  # dimenzije 2x9


def dlt_algoritam(tacke, preslikane_tacke):
    n = len(tacke)
    M = []

    for i in range(n):
        if i > 0:
            Mi = korespodencija(tacke[i], preslikane_tacke[i])
            M = np.concatenate((M, Mi))
        else:
            M = korespodencija(tacke[i], preslikane_tacke[i])

    # SVD => A = U*D*V_tr ; A je dimenzije 2nx9 ; V_tr je dimenzije 9x9
    U, D, V_tr = la.svd(M)

    # Matrica preslikavanja P <==> poslednja vrsta V_tr
    return V_tr[len(V_tr) - 1].reshape(3, 3)


def on_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        unete_koordinate.append([x,y])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ', ' + str(y), (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)   
        
        
#----------------------------------------------------------------------------------------
   

img = cv2.imread("/home/momcilo/PPGR_domaci2/Za slanje br2/4'/prva4.jpeg", cv2.IMREAD_COLOR)
img = cv2.resize(img,(400,500))
cv2.imshow('image', img)
cv2.setMouseCallback('image', on_click)
while True:
    cv2.imshow("image",img)
    if cv2.waitKey(1) & len(unete_koordinate) == 6:
        break

img = cv2.imread("/home/momcilo/PPGR_domaci2/Za slanje br2/4'/druga4.jpeg", 1)
img = cv2.resize(img,dsize=(400,500))
cv2.imshow('image', img)
cv2.setMouseCallback('image', on_click)
while True:
    cv2.imshow("image",img)
    if cv2.waitKey(1) & len(unete_koordinate) == 12:
        break
cv2.destroyAllWindows()


tacke = []
for i in range(6):
    tacke.append(u_homogene(unete_koordinate[i]))

preslikane_tacke = []
for i in range(6):
    unete_koordinate[i+6][0] +=400
    preslikane_tacke.append(u_homogene(unete_koordinate[i+6]))


M = dlt_algoritam(tacke, preslikane_tacke)
M = np.round(M,decimals=10)

img = cv2.imread("/home/momcilo/PPGR_domaci2/Za slanje br2/4'/prva4.jpeg", 1)
img = cv2.resize(img,dsize=(400,500))

img2 = cv2.imread("/home/momcilo/PPGR_domaci2/Za slanje br2/4'/druga4.jpeg", 1)
img2 = cv2.resize(img2,dsize=(400,500))

tmp_img = cv2.imread("/home/momcilo/PPGR_domaci2/Za slanje br2/4'/druga4.jpeg", 1)
tmp_img = cv2.resize(tmp_img,dsize=(400,500))

for i in range(400):
    for j in range(500):          
        tmp_img[j][i] = [0,0,0]
        
im = cv2.hconcat([tmp_img,img2])    
dodatak = cv2.hconcat([img,tmp_img])
M = np.float32(M)
sl = cv2.warpPerspective(dodatak,M,(600,900))
   
for i in range(400):
    for j in range(500): 
        if(im[j,i,0] == 0 and im[j,i,1] == 0 and im[j,i,2] == 0):      
            im[j][i] = sl[j][i]

cv2.imshow('PANORAMA',im)
cv2.waitKey(0)       

imgs = [img,img2]
s = cv2.Stitcher.create()
res = s.stitch(imgs)
