# jMouse--Imaginary-Mouse
[![platform](https://img.shields.io/badge/Implementation-Python-blue.svg)](https://www.android.com)

Controls the mouse action through gestures using Image processing i.e. WebCam only. This innovative tool converts webcam to an imaginary mouse (hence the name). It is a cheap alternative to existing gesture control devices like evo. This project implements correlation-based tracking approach. Move the mouse by tracking a object. By the use of Opencv library in Python i.e. Image processing, your palm will be detected and the mouse cursor will move according to the position of hand.

## Required Libraries

OpenCV(compulsory)<br>
numpy<br>
common<br>
video<br>
win32api, win32con<br>
pyautogui<br>
sys, getopt

## Algorithm outline

![](https://i.imgur.com/AlmKvJ5.png))

1) Firstly, we are detecting the mean value of the selected area by converting frame received by Webcam to a Threshold using Grayscale for precision and Gaussian Blur for removing the noise.
2) After that, we have set the position of the cursor accordingly as the position of hand.
3) For other events to happen, we have detected the tips of the fingers through Edge detection technique.
4) We are processing the number of dots, means if you stick two fingers together that will be a single point.
5) Using this technique and the area of palm, we have implemented some other gesture features.

## Instruction to operate

1) See the count on the top left corner and set a value(count_max).
   When
     count = count_max-1 : LEFT Click
     count = count_max+1 : RIGHT Click
2) Just Place your Hand at the circle shown and press s to start.
3) Keys:
     SPACE    - pause video
     c        - clear targets
     s        - start using as a mouse and set the count
     esc      - close the program
4) Additional features:
     Double Click - By two consecutive left clicks
     Drag - Left Click, hold and Release
     Scroll Down - Reduce hand area and cursor position to lower half of screen  
     Scroll Up - Reduce hand area and cursor position to upper half of screen
     
## Acknowledgements
[Akhil Goel](https://www.facebook.com/goelakhi) for guidance.<br>
[Harshit Kumar](https://github.com/harshit211997), Aman Maghan and Bhavna (Team Members).<br>
[Pranav Mistry’s  Mouseless](http://www.pranavmistry.com/projects/mouseless/)<br>
Google, Stackoverflow, Youtube and OpenCV documentation for solving issues.<br>
www.pythonprogramming.net for tutorials.

## Thanks
©[MIT License](https://github.com/chaira19/jMouse--Imaginary-Mouse/blob/master/LICENSE)<br>
Feel free to contact me for anything related to the repository.
cite @[Chirayu Asati](https://www.quora.com/profile/Chirayu-Asati)
