import numpy as np
import matplotlib.pyplot as plt
import cv2

im = cv2.imread('TawsifExampleDataApr2024\SubsetSmallTiffFiles\img_t1_z1_c1_small_800.tiff', cv2.IMREAD_GRAYSCALE)

im_bw = im.astype(float)
im_bw[(im_bw >= 50) & (im_bw < 60)] = np.nan

vert_wave_pattern = np.nanmean(im_bw, axis=1)

p = np.polyfit(np.arange(len(vert_wave_pattern)), vert_wave_pattern, 2)

corr = np.polyval(p, np.arange(len(vert_wave_pattern))) / vert_wave_pattern

excluded_regions = [
    (0, 0, int(im.shape[0] * 0.1), int(im.shape[1] * 0.1)),
    (int(im.shape[0] * 0.9), 0, im.shape[0], int(im.shape[1] * 0.1)),
    (int(im.shape[0] * 0.9), int(im.shape[1] * 0.9), im.shape[0], im.shape[1])
]

mask = np.ones_like(im_bw)
for region in excluded_regions:
    mask[region[0]:region[2], region[1]:region[3]] = 0

im_bw_corrected = im_bw * corr[:, np.newaxis]
im_bw_corrected[mask == 0] = np.nan
im_bw_corrected[np.isnan(im_bw_corrected)] = 52

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
im_bw_corrected_clahe = clahe.apply(im_bw_corrected.astype(np.uint8))

plt.imshow(im_bw_corrected_clahe, cmap='gray')
plt.show()
