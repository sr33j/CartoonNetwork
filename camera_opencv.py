import cv2
from base_camera import BaseCamera


class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()

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
            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img_cartoon)[1].tobytes()
