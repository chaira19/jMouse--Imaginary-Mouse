#!/usr/bin/env python

'''
MOUSE TRACKING AND GESTURE RECOGNITION(EVO MOUSE)

This project implements correlation-based tracking approach.
Move the mouse by the tracking a object.
Usage:
  --pause  -  Start with playback paused at the first video frame.
              Useful for tracking target selection.

  See the count on the top left corner and set a value(count_max)
  when
  count = count_max-1 : LEFT Click
  count = count_max+1 : RIGHT Click
  Just Place your Hand at the circle shown and press s
Keys:
  SPACE    - pause video
  c        - clear targets
  s        - start using as a mouse and set the count

'''

import numpy as np
import cv2
from common import draw_str, RectSelector
import video
import win32api,win32con
end =0
one=0
def rnd_warp(a):
    h, w = a.shape[:2]
    T = np.zeros((2, 3))
    coef = 0.2
    ang = (np.random.rand()-0.5)*coef
    c, s = np.cos(ang), np.sin(ang)
    T[:2, :2] = [[c,-s], [s, c]]
    T[:2, :2] += (np.random.rand(2, 2) - 0.5)*coef
    c = (w/2, h/2)
    T[:,2] = c - np.dot(T[:2, :2], c)
    return cv2.warpAffine(a, T, (w, h), borderMode = cv2.BORDER_REFLECT)

def divSpec(A, B):
    Ar, Ai = A[...,0], A[...,1]
    Br, Bi = B[...,0], B[...,1]
    C = (Ar+1j*Ai)/(Br+1j*Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C

eps = 1e-5

class MOSSE:
    def __init__(self, frame, rect):
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2-x1, y2-y1])
        x1, y1 = (x1+x2-w)//2, (y1+y2-h)//2
        self.pos = x, y = x1+0.5*(w-1), y1+0.5*(h-1)
        self.size = w, h
        img = cv2.getRectSubPix(frame, (w, h), (x, y))

        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h//2, w//2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()

        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for i in xrange(128):
            a = self.preprocess(rnd_warp(img))
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)
            self.H2 += cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.update_kernel()
        self.update(frame)

    def update(self, frame, rate = 0.125):
        (x, y), (w, h) = self.pos, self.size
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), (x, y))
        img = self.preprocess(img)
        self.last_resp, (dx, dy), self.psr = self.correlate(img)
        self.good = self.psr > 8.0
        if not self.good:
            return

        self.pos = x+dx, y+dy
        self.last_img = img = cv2.getRectSubPix(frame, (w, h), self.pos)
        img = self.preprocess(img)

        A = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        H1 = cv2.mulSpectrums(self.G, A, 0, conjB=True)
        H2 = cv2.mulSpectrums(     A, A, 0, conjB=True)
        self.H1 = self.H1 * (1.0-rate) + H1 * rate
        self.H2 = self.H2 * (1.0-rate) + H2 * rate
        self.update_kernel()

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        h, w = f.shape
        f = np.roll(f, -h//2, 0)
        f = np.roll(f, -w//2, 1)
        kernel = np.uint8( (f-f.min()) / f.ptp()*255 )
        resp = self.last_resp
        resp = np.uint8(np.clip(resp/resp.max(), 0, 1)*255)
        vis = np.hstack([self.last_img, kernel, resp])
        return vis

    def draw_state(self, vis, i):
        (x, y), (w, h) = self.pos, self.size
        x1, y1, x2, y2 = int(x-0.5*w), int(y-0.5*h), int(x+0.5*w), int(y+0.5*h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255))
        if self.good:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
            win32api.SetCursorPos((int(2*x),int(2*y)))
            i=0
            
        else:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(vis, (x2, y1), (x1, y2), (0, 0, 255))
            while(i==0):
                #win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,int(2*x),int(2*y),0,0)
                #win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,int(2*x),int(2*y),0,0)
                i=1
        draw_str(vis, (x1, y2+16), 'PSR: %.2f' % self.psr)

    def preprocess(self, img):
        img = np.log(np.float32(img)+1.0)
        img = (img-img.mean()) / (img.std()+eps)
        return img*self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = resp.shape
        _, mval, _, (mx, my) = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        cv2.rectangle(side_resp, (mx-5, my-5), (mx+5, my+5), 0, -1)
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval-smean) / (sstd+eps)
        return resp, (mx-w//2, my-h//2), psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[...,1] *= -1

class App:
    def __init__(self, video_src, paused = False):
        self.cap = video.create_capture(0)
        _, self.frame = self.cap.read()
        cv2.imshow('frame', self.frame)
        self.rect_sel = RectSelector('frame', self.onrect)
        self.trackers = []
        self.paused = paused

    def onrect(self, rect):
        frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        tracker = MOSSE(frame_gray, rect)
        self.trackers.append(tracker)

    def run(self,one):
        coun_max=0
        while True:
            
            if not self.paused:
                ret, self.frame = self.cap.read()
                if not ret:
                    break
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                blurgray = cv2.GaussianBlur(frame_gray,(5,5),0)
                ret,thresh1 = cv2.threshold(blurgray,70,255,cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
                cv2.imshow('thresh',thresh1)
                image,contours, hierarchy = cv2.findContours(thresh1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                drawing = np.zeros(self.frame.shape,np.uint8)    
                max_area=0
                ci=0
                for i in range(len(contours)):
                        cnt=contours[i]
                        area = cv2.contourArea(cnt)
                        if(area>max_area):
                            max_area=area
                            ci=i
                cnt=contours[ci]
                hull = cv2.convexHull(cnt)
                moments = cv2.moments(cnt)
                if moments['m00']!=0:
                            cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                            cy = int(moments['m01']/moments['m00']) # cy = M01/M00
    
                centr=(cx,cy)       
                cv2.circle(self.frame,centr,5,[0,0,255],2)
                cv2.drawContours(drawing,[cnt],0,(0,255,0),2) 
                cv2.drawContours(drawing,[hull],0,(0,0,255),2) 
          
                cnt = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                hull = cv2.convexHull(cnt,returnPoints = False)
                
                if(1):
                           defects = cv2.convexityDefects(cnt,hull)
                           mind=0
                           maxd=0
                           coun=0
                           
                           for i in range(defects.shape[0]):
                                coun=coun+1
                                s,e,f,d = defects[i,0]
                                start = tuple(cnt[s][0])
                                end = tuple(cnt[e][0])
                                far = tuple(cnt[f][0])
                                dist = cv2.pointPolygonTest(cnt,centr,True)
                                cv2.line(self.frame,start,end,[0,255,0],2)
                                cv2.circle(self.frame,far,5,[0,0,255],-1)
                                cv2.circle(self.frame,start,5,[255,0,0],-1)
                    
                           i=0
                           font = cv2.FONT_HERSHEY_SIMPLEX
                           cv2.putText(self.frame,str(coun),(0,40), font, 1,(0,0,0),2)
                           cv2.putText(self.frame,str(coun_max),(0,80), font, 1,(0,0,0),2)
                          
                           if(coun == coun_max+1):
                               while(one==0):
                                   (x,y) = win32api.GetCursorPos()
                                   win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN,int(2*x),int(2*y),0,0)
                                   win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP,int(2*x),int(2*y),0,0)
                                   end = 1
                                   one=1
                           if(coun == coun_max-1):
                               while(one==0):
                                   (x,y) = win32api.GetCursorPos()
                                   win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,int(2*x),int(2*y),0,0)
                                   win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,int(2*x),int(2*y),0,0)
                                   end = 1
                                   one=1
                                   
                           if(coun == coun_max):
                               end = 0
                               one=0
                for tracker in self.trackers:
                    tracker.update(frame_gray)

            vis = self.frame.copy()
            for tracker in self.trackers:
                tracker.draw_state(vis,0)
            if len(self.trackers) > 0:
                cv2.imshow('tracker state', self.trackers[-1].state_vis)
            self.rect_sel.draw(vis)

            cv2.imshow('frame', vis)
            ch = cv2.waitKey(10)
            if ch == ord('s'):
                coun_max = coun
                a=cx-80
                b=cy-80
                c=cx+80
                d=cy+80
                frame_gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                tracker = MOSSE(frame_gray,(a,b,c,d))
                self.trackers.append(tracker)
            if ch == 27:
                self.cap.release()
                break
            if ch == ord(' '):
                self.paused = not self.paused
            if ch == ord('c'):
                self.trackers = []
                coun_max = 0
            


print __doc__
import sys, getopt
opts, args = getopt.getopt(sys.argv[1:], '', ['pause'])
opts = dict(opts)
try: video_src = args[0]
except: video_src = '0'
App(video_src, paused = '--pause' in opts).run(one)
cv2.destroyAllWindows()
