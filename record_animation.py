import numpy as np
import cv2

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter('output.mp4', fourcc, 10.0, (width, height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # frame = cv2.flip(frame,0)

        # _, img = camera.read()
        img = frame

        num_down = 2       # number of downsampling steps
        num_bilateral = 7  # number of bilateral filtering steps
         
        # img_rgb = cv2.imread("img_example.jpg")
         
        # downsample image using Gaussian pyramid
        img_color = img
        for _ in range(num_down):
            img_color = cv2.pyrDown(img_color)
         
        # repeatedly apply small bilateral filter instead of
        # applying one large filter
        for _ in range(num_bilateral):
            img_color = cv2.bilateralFilter(img_color, d=9,
                                            sigmaColor=9,
                                            sigmaSpace=7)
     
        # upsample image to original size
        for _ in range(num_down):
            img_color = cv2.pyrUp(img_color)

        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)

        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY,
                                     blockSize=9,
                                     C=3)

        # convert back to color, bit-AND with color image
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        img_cartoon = cv2.bitwise_and(img_color, img_edge)

        # write the flipped frame
        out.write(img_cartoon)

        cv2.imshow('frame',img_cartoon)
        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
            break
    else:
        break

# Release everything if job is finished
out.release()
cap.release()
cv2.destroyAllWindows()