import cv2
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 7))
figR = 3
figC = 3

def sinusFilter(img):
    try:
        imgFft = np.fft.fft2(img)
        imgFftS = np.fft.fftshift(imgFft)
        magnitude_spectrum_before = np.log(1+np.abs(imgFftS))

        rows, cols = img.shape
        mask = np.zeros((rows, cols), np.uint8)
        mask[0:rows, 0:cols] = 1
        center_row, center_col = int(rows/2), int(cols/2)
        mask[center_row, center_col-9:center_col-7] = 0
        mask[center_row, center_col+7:center_col+9] = 0

        imgFftSConvMask = imgFftS*mask
        magnitude_spectrum_after = magnitude_spectrum_before * mask

        imgIfftS = np.fft.ifftshift(imgFftSConvMask)
        imgSinusFilter = np.abs(np.fft.ifft2(imgIfftS))
        imgSinusFilter = imgSinusFilter.astype(np.uint8)

        fig.add_subplot(figR, figC, 1), plt.imshow(img, cmap='gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        fig.add_subplot(figR, figC, 2), plt.imshow(
            magnitude_spectrum_before, cmap='gray')
        plt.title('Magnitude Spectrum Before'), plt.xticks([]), plt.yticks([])
        fig.add_subplot(figR, figC, 3), plt.imshow(
            magnitude_spectrum_after, cmap='gray')
        plt.title('Magnitude Spectrum After'), plt.xticks([]), plt.yticks([])
        fig.add_subplot(figR, figC, 4), plt.imshow(imgSinusFilter, cmap='gray')
        plt.title('Inverse'), plt.xticks([]), plt.yticks([])
        imgSinusFilter = imgSinusFilter[0:460, 0:460]
        return imgSinusFilter

    except Exception as e:
        print("Error", e)

path = r'images-Projects/origin.png'
img = cv2.imread(path)

if img is None:
    print("check path")
    raise
try:
    # Chuyển ảnh đa mức xám 
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(imgGray.shape)

    # Lọc nhiễu Sin
    imgSinusFilter = sinusFilter(imgGray)

    # Lọc nhiễu muối tiêu lần 1
    imgMedianFilter = cv2.medianBlur(imgSinusFilter, 5)
    fig.add_subplot(figR, figC, 5), plt.imshow(imgMedianFilter, cmap='gray')
    plt.title('Median'), plt.xticks([]), plt.yticks([])

    # Lọc nhiễu ánh sáng bằng ngưỡng thích nghi
    imgAdaptiveThreshold = cv2.adaptiveThreshold(
        imgMedianFilter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 125, 0)
    fig.add_subplot(figR, figC, 6), plt.imshow(imgAdaptiveThreshold, cmap='gray')
    plt.title('Adaptive Threshold 2'), plt.xticks([]), plt.yticks([])

    # Lọc nhiễu muối tiêu lần 2
    imgMedianFilter2 = cv2.medianBlur(imgAdaptiveThreshold, 7)
    fig.add_subplot(figR, figC, 7), plt.imshow(imgMedianFilter2, cmap='gray')
    plt.title('Median 2'), plt.xticks([]), plt.yticks([])
     
    # Đếm số vật thể
    contours, hierarchy = cv2.findContours(
        imgMedianFilter2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rgb = imgMedianFilter2.copy()
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 1)

    fig.add_subplot(figR, figC, 8), plt.imshow(rgb)
    plt.title('Contour Image'), plt.xticks([]), plt.yticks([])

    print("Number of object", len(contours))
    plt.show()

except Exception as e:
    print("Error", e)
cv2.waitKey()
