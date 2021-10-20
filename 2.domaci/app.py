from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import cv2
import numpy as np
from scipy.linalg import svd
import sys

size = (800, 600)
dots = []
cv_image = None
canvas = None
frame = None
algorithm_started = False
img = None

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


def create_circle(x, y, r, canvasName):
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    dots.append([x, y, 1])
    return canvasName.create_oval(x0, y0, x1, y1)    
    
def start_algorithm():
    global dots, cv_image, canvas, img
    
    sorted_dots(dots)
    print(f'Dots: {dots}')
    
    prim_dots = []
    prim_dots.append(dots[0]) #prvu tacku fiksiramo
    prim_dots.append([dots[0][0], dots[1][1], 1]) #druga ima svoju visinu ali x koordinatu kao prva tacka
    prim_dots.append([dots[2][0], dots[1][1], 1]) #treca ima visinu druge tacke i svoju x koordinatu
    prim_dots.append([dots[2][0], dots[0][1], 1]) #cetvrta ima visinu prve tacke i x koordinatu kao treca tacka
    print(f'Dots\': {prim_dots}')

    zip_points = list(zip(dots, prim_dots))
    PDLT = DLT_algorithm(zip_points)
    
    img_output = cv2.warpPerspective(cv_image, PDLT, size)
    img_output = cv2.cvtColor(img_output, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_output)
    img = ImageTk.PhotoImage(img)
    
    canvas.create_image(0,0,image=img,anchor="nw")
    
    
    
if __name__ == "__main__":
    
    root = Tk()
    root.geometry("1080x800+200+200")
    root.resizable(width=True, height=True)
    
    File = filedialog.askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
    if File[-4:] != ".bmp":
        print("Slika mora biti bmp format")
        sys.exit(0) 
    
    cv_image = cv2.imread(File)    
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)

    #postavljanja canvasa
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, width=800, height=800)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))
    
    
    def printcoords(event):
        global algorithm_started
        if len(dots) <= 3:
            create_circle(event.x, event.y, 4, canvas)
        if len(dots) == 4 and not algorithm_started:
            algorithm_started = True
            start_algorithm()
            
    
    #funkcija koja se poziva kad se klikne na fotografiju
    canvas.bind("<Button 1>", printcoords)
    root.mainloop()
    
