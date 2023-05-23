import cv2
import numpy as np

img = cv2 . imread ( ' norway . jpg ' )
rows , cols = img . shape [ : 2 ]

kernel_x = cv2 . getGausianKernel(cols,200)
kernel_y = cv2 . getGausianKernel(rows,200)

kernel = kernel_y * kernel_x.T
kernel = kernel /np.linalg.norm( kernel )

mask = 255 * kernel
output = np . copy ( img )

for i in range ( 3 ) :
    output [ : , : , i ] = output [ : , : , i ] * mask
    cv2.imshow('Original',img )
    cv2.imshow('Gaussian',output )
    cv2.waitKey(0) 