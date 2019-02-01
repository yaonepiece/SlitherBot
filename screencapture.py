import numpy as np
from PIL import ImageGrab
import cv2, os, time, select

class ScreenCapturer():
	def __init__(self,x,y,cropx,cropy,outx=0,outy=0):
		'''
		(x,y): center pixel of the capture according to the computer screen
		(cropx,cropy): size of the captured picture
		(outx,outy): size of the output from auto encoder, default to (cropx,cropy)
		'''
		self.x=x
		self.y=y
		self.outx=self.cropx=cropx
		self.outy=self.cropy=cropy
		if outx>0:
			self.outx=outx
		if outy>0:
			self.outy=outy
		
		self.data_ready=False
		self.screen_data=np.ndarray((4,self.outx,self.outy),dtype='uint8')
		self.screen_cnt=0
	
	def __crop_center(self,img,cropx,cropy):
		startx = self.x-(cropx//2)
		starty = self.y-(cropy//2)
		return img[starty:starty+cropy,startx:startx+cropx]
	
	def get_pic(self):
		screen_pil = ImageGrab.grab()
		screen_np = np.asarray(screen_pil,dtype='uint8')
		screen_np = screen_np[:,:,[2,1,0]]
		screen_np = self.__crop_center(screen_np,self.cropx,self.cropy)
		return screen_np
	
	def get_gray(self):
		screen_pil = ImageGrab.grab()
		screen_np = np.asarray(screen_pil,dtype='uint8')
		screen_np = screen_np[:,:,[2,1,0]]
		screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)
		screen_np = self.__crop_center(screen_np,self.cropx,self.cropy)
		return screen_np
		
	def save_pic(self,screen):
		self.screen_data[self.screen_cnt]=screen
		self.screen_cnt+=1
		if self.screen_cnt==4:
			self.screen_cnt=0
			self.data_ready=True

if __name__ == '__main__':
	t = time.time()
	count = 0
	scap=ScreenCapturer(960,540,64,64)

	while(True):
		if(time.time() - t > 1):
			os.system("cls")
			print(count)
			t = time.time()
			count = 0
		screen=scap.get_pic()
		cv2.imshow('window1',screen)
		if cv2.waitKey(25) & 0xFF == ord('q'):
			cv2.destroyAllWindows()
			break
		count += 1
		# print(select.poll())
	