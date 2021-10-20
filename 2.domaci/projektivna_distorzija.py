
import cv2
import numpy as np
from scipy.linalg import svd

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

def sorted_dots(dots):
    #Lista tacaka nam je uvek u obliku:
    # X1----X4
    # |      | 
    # X2----X3
    dots = sorted(dots, key=lambda dot: dot[0])
    print(dots)
    if dots[1][1] <= dots[0][1]:
        tmp = dots[0]
        dots[0] = dots[1]
        dots[1] = tmp
    if dots[3][1] >= dots[2][1]:
        tmp = dots[3]
        dots[3] = dots[2]
        dots[2] = tmp

    return dots

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),3,(255,0,0),-1)
        dots.append([x, y, 1])

dots = []

if __name__ == '__main__':
    rows, cols = (600,600)

    img = cv2.imread('photo-building.jpg')
    img = cv2.resize(img,(800,600))
    cv2.imshow('image',img)
    cv2.setMouseCallback('image' ,draw_circle)

    while len(dots) < 4:
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    dots = sorted_dots(dots)
    print(f'Dots: {dots}')
    prim_dots = []
    prim_dots.append(dots[0]) #prvu tacku fiksiramo
    prim_dots.append([dots[0][0], dots[1][1], 1]) #druga ima svoju visinu ali x koordinatu kao prva tacka
    prim_dots.append([dots[2][0], dots[1][1], 1]) #treca ima visinu druge tacke i svoju x koordinatu
    prim_dots.append([dots[2][0], dots[0][1], 1]) #cetvrta ima visinu prve tacke i x koordinatu kao treca tacka

    print(f'Dots\': {prim_dots}')

    zip_points = list(zip(dots, prim_dots))
    PDLT = DLT_algorithm(zip_points)

    img_output = cv2.warpPerspective(img, PDLT, (800, 600))

    cv2.imshow('Output', img_output)
    cv2.waitKey()
