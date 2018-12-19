import os
import cv2
import time
import numpy as np
from mss import mss


class ScreenCapturer:
    def __init__(self, x, y, cropx, cropy, outx=0, outy=0, n=4):
        # (x,y): top left pixel of the capture according to the computer screen
        # (cropx,cropy): size of the captured picture
        # (outx,outy): size of the output from auto encoder, default to (cropx,cropy)
        self.mss = mss()
        self.x = x
        self.y = y
        self.outx = outx if outx else cropx
        self.outy = outy if outy else cropy
        self.area = {"left": x, "top": y, "width": cropx, "height": cropy}
        self.stack_size = n
        self.data_ready = False
        self.screen_cnt = 0
        self.screen_data = np.ndarray((self.stack_size, self.outx, self.outy), dtype='uint8')

    def get_pic(self):
        im = np.asarray(self.mss.grab(self.area), dtype='uint8')
        return im[:, :, :3]

    def get_gray(self):
        im = np.asarray(self.mss.grab(self.area), dtype='uint8')
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
        print(im.shape)
        return im

    def save_pic(self):
        self.screen_data[:-1] = self.screen_data[1:]
        self.screen_data[-1] = self.get_gray()
        if self.screen_cnt < self.stack_size:
            self.screen_cnt += 1
        if self.screen_cnt == self.stack_size:
            self.data_ready = True

    def get_batch(self):
        return self.screen_data

def main():
    t = time.time()
    count = 0
    scap = ScreenCapturer(40, 40, 512, 512)

    while True:
        if (time.time() - t) > 1:
            os.system("clear")
            print(count)
            t = time.time()
            count = 0
        screen = scap.get_pic()
        cv2.imshow('window1', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        count += 1


if __name__ == '__main__':
    main()
