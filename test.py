#import required libraries
import numpy as np
import cv2
import win32api,win32con
import pyautogui

def click(x,y):
      win32api.SetCursorPos((x,y))
      
# function to draw rectangle
def draw_rectangle(event, x, y, flags, params):
    global x_init, y_init, drawing, top_left_pt, bottom_right_pt,top,bottom,track_window

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_init, y_init = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            top_left_pt = (min(x_init, x), min(y_init, y))
            bottom_right_pt = (max(x_init, x), max(y_init, y))
            img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        top_left_pt = (min(x_init, x), min(y_init, y))
        bottom_right_pt = (max(x_init, x), max(y_init, y))
        img[y_init:y, x_init:x] = 255 - img[y_init:y, x_init:x]
        top,bottom = top_left_pt,bottom_right_pt
        (r,h),(c,w) = top_left_pt,bottom_right_pt
        track_window = (r,h,c-r,w-h)

if __name__=='__main__':
    drawing= False
    top_left_pt, bottom_right_pt=(-1,-1), (-1,-1)



    
cap = cv2.VideoCapture(0)
cv2.namedWindow('Webcam')
cv2.setMouseCallback('Webcam', draw_rectangle)
while True:
          ret, frame =cap.read()
          #img=cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
          (r,h),(c,w) = top_left_pt,bottom_right_pt
          img = cv2.rectangle(frame, (r,h), (c,w), 255,2)
          cv2.imshow('Webcam',img)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
     
     # take first frame of the video
ret,frame = cap.read()

     # setup initial location of window
#(r,h),(c,w) = top_left_pt,bottom_right_pt  # simply hardcoded the values
#track_window = (r,h,c-r,w-h)

    # set up the ROI for tracking
roi = frame[r:c,h:w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    
    # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
top,bottom = top_left_pt,bottom_right_pt
while(1):
    ret ,frame = cap.read()
    
    if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
            # apply meanshift to get the new location
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    
            # Draw it on image
            x,y,w,h = track_window
            img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
            if (top != top_left_pt) & (bottom != bottom_right_pt):
                (r,e),(c,t) = top_left_pt,bottom_right_pt
                img = cv2.rectangle(img2, (r,e), (c,t), 255,2)
                cv2.imshow('Webcam',img)
                pyautogui.moveTo(10,10)
            else:
                cv2.imshow('Webcam',img2)
                
            k = cv2.waitKey(60) & 0xff

            #click(x,y)
            if k == 27:
                break
            else:
                cv2.imwrite(chr(k)+".jpg",img2)
    
    else:
            break


#class added        


#click(10,10)    
cv2.destroyAllWindows()
cap.release()
