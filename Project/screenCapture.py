import os
import cv2
import time
import numpy as np
from mss import mss


class ScreenCapturer:
    def __init__(self, x, y, cropx, cropy, outx=0, outy=0, n=4):
        # (x,y): center pixel of the capture according to the computer screen
        # (cropx,cropy): size of the captured picture
        # (outx,outy): size of the output from auto encoder, default to (cropx,cropy)
        self.mss = mss()
        self.x = x - cropx//2
        self.y = y - cropy//2
        self.outx = outx if outx else cropx
        self.outy = outy if outy else cropy
        self.area = {"left": self.x, "top": self.y, "width": cropx, "height": cropy}
        if cropx != outx or cropy != outy:
            self.resize=True
        else:
            self.resize=False
        self.stack_size = n
        self.data_ready = False
        self.screen_data = np.ndarray((0, self.outx, self.outy), dtype='float16')

    def get_pic(self):
        im = np.asarray(self.mss.grab(self.area), dtype='float16')/256
        if self.resize:
            im = cv2.resize(im, (self.outx, self.outy))
        return im[:, :, :3]

    def get_gray(self):
        im = np.asarray(self.mss.grab(self.area), dtype='float16')/256
        if self.resize:
            im = cv2.resize(im, (self.outx,self.outy))
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY)
        return im

    def save_pic(self, screen):
        if len(self.screen_data) == self.stack_size:
            self.screen_data = self.screen_data[1:]
        self.screen_data = np.append(self.screen_data, [screen], axis=0)
        if len(self.screen_data) == self.stack_size:
            self.data_ready = True
    
    def clear(self):
        self.screen_data=np.ndarray((0, self.outx, self.outy), dtype='float16')
        self.data_ready=False


def main():
    t = time.time()
    count = 0
    scap = ScreenCapturer(960, 512, 1024, 512, 256, 128)

    while True:
        if (time.time() - t) > 1:
            os.system("cls")
            print(count)
            t = time.time()
            count = 0
        screen = scap.get_gray()
        cv2.imshow('window1', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        count += 1


if __name__ == '__main__':
    main()
