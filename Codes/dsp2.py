import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('Elon.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Apply 2D DFT
img1_dft = np.fft.fft2(img1)
img1_dft_shift = np.fft.fftshift(img1_dft)

# Calculate the magnitude spectrum
img1_mag = np.abs(img1_dft_shift)

# Display the original image and the magnitude spectrum
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(img1, cmap='gray')
axs[0].set_title('Original Image')

axs[1].imshow(np.log10(1+img1_mag), cmap='gray')
axs[1].set_title('Magnitude Spectrum (log scale)')
# plt.show()

# Apply inverse 2D DFT
img1_idft= np.fft.ifft2(img1_dft)

# Take the real part of the reconstructed image
img1_idft= np.real(img1_idft)

# Display the original image and the reconstructed image
axs[2].imshow((img1_idft), cmap='gray')
axs[2].set_title('Reconstructed Image')
plt.show()