from kivy.logger import Logger
import logging
Logger.setLevel(logging.TRACE)
from playsound import playsound
from kivy.app import App
import kivy
from kivy.lang import Builder
from kivy.uix.widget import  Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
import numpy as np
import kivy.graphics.texture 
from cv2 import cv2
import time
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from datetime import datetime
from kivy.config import Config
Config.set('graphics','width','360')
Config.set('graphics','high','550')

finding_eye=True
getting_mid=False
ex_1=False
look_right=False
sta=time.perf_counter()
mid_l=(0,0)
mid_r=(0,0)
lastkp_l=(0,0)
lastkp_r=(0,0)
thresh_l=40
thresh_r=40
where=0
class eye:
    ld=(250,250)
    ur=(250,250)
    mid=(250,250)
    def __init__(self  ,which):  
        self.which=which     
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.filterByArea = True
        detector_params.maxArea = 300
        self.detector = cv2.SimpleBlobDetector_create(detector_params)
    
    def update(self,ld,ur):
        self.ur=ur
        self.ld=ld

    def middle(self,xy):
        self.mid=xy
    
    def get_big_pos(self,frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f=self.face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        if (len(f)!=0):
            f=f[0]
            if (self.which=='l'):
                half_face=np.copy(frame[f[1]:f[1]+f[3],f[0]:f[0]+f[2]//2])
            if (self.which=='r'):
                half_face=np.copy(frame[f[1]:f[1]+f[3],f[0]+f[2]//2:f[0]+f[2]])
            if (len(frame)!=0):
                gray_frame = cv2.cvtColor(half_face, cv2.COLOR_BGR2GRAY)
                ey=self.eye_cascade.detectMultiScale(gray_frame, 1.3, 5    )
                if (len(ey)!=0):
                    ey=ey[0]
                    if (self.which=='l'):
                        self.update( (f[0]+ey[0],f[1]+ey[1]) , (f[0]+ey[0]+ey[2],f[1]+ey[1]+ey[3]) )
                    if (self.which=='r'):
                        self.update( (f[2]//2+f[0]+ey[0],f[1]+ey[1]) , (f[2]//2+f[0]+ey[0]+ey[2],f[1]+ey[1]+ey[3]) )
                    return True,half_face
                
        return False,frame


    def getpos(self,frame):
        if ( self.ld[1]==250):
            return [],frame
        img=np.copy(frame)
        img=img[ self.ld[1]+12: self.ur[1] , self.ld[0]  : self.ur[0] ]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if (self.which=='l'):
            _, img = cv2.threshold(img, thresh_l  , 255, cv2.THRESH_BINARY)
        if (self.which=='r'):
            _, img = cv2.threshold(img, thresh_r  , 255, cv2.THRESH_BINARY)
        img = cv2.erode(img, None, iterations=2) #1
        img = cv2.dilate(img, None, iterations=4) #2
        img = cv2.medianBlur(img, 5) #3
        
        img2=np.copy(img)
        
        h, w = img.shape
        keypoints =self.detector.detect(img)
        return keypoints,img2    
def nothing(x):
    pass
cv2.namedWindow('image3') 
cv2.namedWindow('image')  
cv2.createTrackbar('tresh_r', 'image3', 0, 255, nothing)
cv2.createTrackbar('tresh_l', 'image3', 0, 255, nothing)
cv2.namedWindow('image2')  
el=eye('l')
er=eye('r')


def find_eyes(frame):
    frame = cv2.rectangle(frame,el.ld,el.ur,(0,255,0),1)
    frame = cv2.rectangle(frame,er.ld,er.ur,(0,255,0),1)
    return frame

def do_frame(frame):

    global finding_eye
    global getting_mid
    global ex_1
    global mid_l
    global mid_r
    global lastkp_l
    global lastkp_r
    global look_right
    global thresh_l
    global thresh_r
    global where
    thresh_l = cv2.getTrackbarPos('tresh_l', 'image3')
    thresh_r = cv2.getTrackbarPos('tresh_r', 'image3')
    if (finding_eye==True):
        el.get_big_pos(frame)
        er.get_big_pos(frame)
    frame=find_eyes(frame)
    kp_l,img=el.getpos(frame) 
    kp_r,img2=er.getpos(frame)
    if ((kp_r!=[]) ) :
        lastkp_r=kp_r[0]
        #print(kp_r)
    if (kp_l!=[]):
        lastkp_l=kp_l[0]
        #print("left",kp_l)
    cv2.imshow('image',img)
    cv2.imshow('image2',img2)
    print(where)
    if (getting_mid==True) :
        if ((lastkp_r!=[]) & (lastkp_l!=[])) :
            mid_r=lastkp_r
            mid_l=lastkp_l
            #print("r",mid_r.pt)
            #print("l",mid_l.pt)
            getting_mid=False
            if (where ==0):
                playsound('look to the right and, press the start button.mp3',False)  

            if (where  == 1):
                playsound('look to the left and, press the start button (1).mp3',False)  
   
    if  ((look_right==True) & (where==0) ) :
        #print ("diff",lastkp_l.pt[0]-mid_l.pt[0])
        print("sta",(time.perf_counter()-sta))
        if (  time.perf_counter()-sta  < 0.1):
            playsound('five. four. three. two. one..mp3',False)  
        if (lastkp_l.pt[0]-mid_l.pt[0]  > 0 ):
            playsound('oops.mp3',False)  
            look_right=False
        if ( time.perf_counter()-sta > 5):
            playsound('great job.mp3',False)
            look_right=False
            where=1
    elif  ((look_right==True) & (where==1) ) :
        #print ("diff",lastkp_l.pt[0]-mid_l.pt[0])
        print("sta",(time.perf_counter()-sta))
        if (  time.perf_counter()-sta  < 0.1):
            playsound('five. four. three. two. one..mp3',False)  
        if (lastkp_r.pt[0]-mid_r.pt[0]  < 0 ):
            playsound('oops.mp3',False)  
            look_right=False
        if ( time.perf_counter()-sta > 5):
            playsound('great job.mp3',False)
            look_right=False
            where=0
    else :
        print("geot")
    return frame




class MyCamera(Camera):
    def __init__(self, **kwargs):
        super(MyCamera, self).__init__(**kwargs)

    def _camera_loaded(self, *largs):
        if kivy.platform == 'android':
            self.texture = kivy.graphics.texture.Texture.create(size=self.resolution, colorfmt='bgr')
            self.texture_size = list(self.texture.size)
        else:
            self.texture = self._camera.texture
            self.texture_size = list(self.texture.size)

    def on_tex(self, *l):
        if kivy.platform == 'android':
            buf = self._camera.grab_frame()
            if buf is None:
                return
            frame = self._camera.decode_frame(buf)
        else:
            ret, frame = self._camera._device.read()
        if frame is None:
            print("No")

        buf = self.process_frame(frame)
        self.texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        super(MyCamera, self).on_tex(*l)

    def process_frame(self,frame):
        frame=do_frame(frame)
        return frame.tobytes()
    def on_touch_down(self,touch):
        pos=((touch.pos[0]/640-0.128))*640,((1-(touch.pos[1]-60)/480))*480



class CameraClick(Screen,BoxLayout):
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        global finding_eye
        finding_eye=not finding_eye
    def capture2(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        global getting_mid
        getting_mid=True
    def capture3(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        global look_right
        global sta
        sta=time.perf_counter()
        look_right=True    



class MainWindow(Screen):
    pass


class WindowManager(ScreenManager):
    pass


kv = Builder.load_file("my.kv")


class MyMainApp(App):
    def build(self):
        return kv


if __name__ == "__main__":
    MyMainApp().run()


