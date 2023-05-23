import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image and convert it to grayscale
img1 = cv2.imread('Elon.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Compute 1D DFT of rows
img1_dft_rows = np.fft.fft(img1_gray, axis=1)
img1_dft_rows_shift = np.fft.fftshift(img1_dft_rows)

# Compute 1D DFT of columns
img1_dft_cols = np.fft.fft(img1_gray, axis=0)
img1_dft_cols_shift = np.fft.fftshift(img1_dft_cols)

fig, axs = plt.subplots(1, 4, figsize=(10, 5))
axs[0].imshow(img1_gray, cmap='gray')
axs[0].set_title('Original Image')

# Visualize the magnitude spectrum of the DFT of a row
axs[1].imshow(np.log10(1+np.abs(img1_dft_rows_shift)), cmap='gray')
axs[1].set_title('Dft on rows')

# Visualize the magnitude spectrum of the DFT of a column
axs[2].imshow(np.log10(1+np.abs(img1_dft_cols_shift)), cmap='gray')
axs[2].set_title('Dft on columns')


img1_idft_rows=np.fft.ifft(img1_dft_rows,axis=1)
img1_idft_cols=np.fft.ifft(img1_dft_cols,axis=0)

axs[3].imshow(np.real(img1_idft_rows+img1_idft_cols), cmap='gray')
axs[3].set_title('Reconstructed image')
plt.show()