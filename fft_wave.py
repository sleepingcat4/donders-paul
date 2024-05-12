import cv2
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------
def notch_reject_filter(shape, d0=9, u_k=0, v_k=0):
    P, Q = shape
    # Initialize filter with zeros
    H = np.zeros((P, Q))

    # Traverse through filter
    for u in range(0, P):
        for v in range(0, Q):
            # Get euclidean distance from point D(u,v) to the center
            D_uv = np.sqrt((u - P / 2 + u_k) ** 2 + (v - Q / 2 + v_k) ** 2)
            D_muv = np.sqrt((u - P / 2 - u_k) ** 2 + (v - Q / 2 - v_k) ** 2)

            if D_uv <= d0 or D_muv <= d0:
                H[u, v] = 0.0
            else:
                H[u, v] = 1.0

    return H

img = cv2.imread('TawsifExampleDataApr2024\SubsetSmallTiffFiles\img_t1_z1_c1_small_800.tiff', 0)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
phase_spectrumR = np.angle(fshift)
magnitude_spectrum = 20*np.log(np.abs(fshift))

img_shape = img.shape

NotchFilter = notch_reject_filter(img_shape, 4, 10, 0)


NotchRejectCenter = fshift * NotchFilter 
NotchReject = np.fft.ifftshift(NotchRejectCenter)
# Compute the inverse DFT of the result
inverse_NotchReject = np.fft.ifft2(NotchReject)  


Result = np.abs(inverse_NotchReject)

plt.imshow(Result, cmap='gray')
plt.show()

cv2.imwrite('image_filtered.png', Result)
cv2.imwrite('notch_filter.png', NotchFilter*255)
cv2.imwrite('frequency_domain.png', magnitude_spectrum)