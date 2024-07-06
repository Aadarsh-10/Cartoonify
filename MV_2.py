# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:07:14 2024

@author: aadar
"""



import cv2
import easygui
import numpy as np
import imageio
import sys
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image


top=tk.Tk()
top.geometry('400x400')
top.title('Image to cartoon')
top.configure(background='black')
label=Label(top,background='white', font=('calibri',20,'bold'))


def upload():
    ImagePath=easygui.fileopenbox()
    cartoonify(ImagePath)


def cartoonify(ImagePath):
    originalimage = cv2.imread(ImagePath)
    
    originalimage_gb = cv2.GaussianBlur(originalimage, (7, 7) ,0)
    originalimage_mb = cv2.medianBlur(originalimage_gb, 5)
    originalimage_bf = cv2.bilateralFilter(originalimage_mb, 5, 80, 80)

    originalimage_lp_im = cv2.Laplacian(originalimage, cv2.CV_8U, ksize=5)
    originalimage_lp_gb = cv2.Laplacian(originalimage_gb, cv2.CV_8U, ksize=5)
    originalimage_lp_mb = cv2.Laplacian(originalimage_mb, cv2.CV_8U, ksize=5)
    originalimage_lp_al = cv2.Laplacian(originalimage_bf, cv2.CV_8U, ksize=5)

    originalimage_lp_im_grey = cv2.cvtColor(originalimage_lp_im, cv2.COLOR_BGR2GRAY)
    originalimage_lp_gb_grey = cv2.cvtColor(originalimage_lp_gb, cv2.COLOR_BGR2GRAY)
    originalimage_lp_mb_grey = cv2.cvtColor(originalimage_lp_mb, cv2.COLOR_BGR2GRAY)
    originalimage_lp_al_grey = cv2.cvtColor(originalimage_lp_al, cv2.COLOR_BGR2GRAY)

    _, EdgeImage = cv2.threshold(originalimage, 127, 255, cv2.THRESH_BINARY)

    blur_im = cv2.GaussianBlur(originalimage_lp_im_grey, (5, 5), 0)
    blur_gb = cv2.GaussianBlur(originalimage_lp_gb_grey, (5, 5), 0)
    blur_mb = cv2.GaussianBlur(originalimage_lp_mb_grey, (5, 5), 0)
    blur_al = cv2.GaussianBlur(originalimage_lp_al_grey, (5, 5), 0)

    _, tresh_im = cv2.threshold(blur_im, 245, 255,cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
    _, tresh_gb = cv2.threshold(blur_gb, 245, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, tresh_mb = cv2.threshold(blur_mb, 245, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, tresh_al = cv2.threshold(blur_al, 245, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    inverted_original = cv2.subtract(255, tresh_im)
    inverted_GaussianBlur = cv2.subtract(255, tresh_gb)
    inverted_MedianBlur = cv2.subtract(255, tresh_mb)
    inverted_Bilateral = cv2.subtract(255, tresh_al)

    originalimage_reshaped = originalimage.reshape((-1,3))

    originalimage_reshaped = np.float32(originalimage_reshaped)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = 8

    _, label, center = cv2.kmeans(originalimage_reshaped, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]

    originalimage_Kmeans = res.reshape((originalimage.shape))

    div = 64
    originalimage_bins = originalimage // div * div + div // 2

    inverted_Bilateral = cv2.cvtColor(inverted_Bilateral, cv2.COLOR_GRAY2RGB)

    cartoon_Bilateral = cv2.bitwise_and(inverted_Bilateral, originalimage_bins)

    cv2.imwrite('C:/Users/aadar/Downloads/CartoonImage.png', cartoon_Bilateral)



    
    originalimage = cv2.cvtColor(originalimage, cv2.COLOR_BGR2RGB)
    
    if originalimage is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()

    ReSized1 = cv2.resize(originalimage, (940,610))
    
    grayScaleImage= cv2.cvtColor(originalimage, cv2.COLOR_BGR2GRAY)
    ReSized2 = cv2.resize(grayScaleImage, (940,610))

    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    ReSized3 = cv2.resize(smoothGrayScale, (940,610))

    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 9)

    ReSized4 = cv2.resize(getEdge, (940,610))
   
   
    colorImage = cv2.bilateralFilter(originalimage, 9, 300, 300)
    ReSized5 = cv2.resize(colorImage, (940,610))
  
    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)

    ReSized6 = cv2.resize(cartoonImage, (940,610))
   
    images=[ReSized1, ReSized2, ReSized3, ReSized4, ReSized5, ReSized6]

    fig, axes = plt.subplots(3,2, figsize=(8,8), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')

    save1=Button(top,text="Save image",command=lambda: save(cartoon_Bilateral, ImagePath),padx=30,pady=5)
    save1.configure(background='white', foreground='black',font=('calibri',10,'bold'))
    save1.pack(side=TOP,pady=50)
    
    plt.show()



def save(ReSized6, ImagePath):
    
    newName="cartoon_Image"
    path1 = os.path.dirname(ImagePath)
    extension=os.path.splitext(ImagePath)[1]
    path = os.path.join(path1, newName+extension)
    cv2.imwrite(path, cv2.cvtColor(ReSized6, cv2.COLOR_RGB2BGR))
    I= "Image saved by name " + newName +" at "+ path
    tk.messagebox.showinfo(title=None, message=I)



upload=Button(top,text="Cartoonify",command=upload,padx=10,pady=5)
upload.configure(background='white', foreground='black',font=('calibri',10,'bold'))
upload.pack(side=TOP,pady=50)


top.mainloop()
