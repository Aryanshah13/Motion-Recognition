#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

motion_detected = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    fgmask = fgbg.apply(frame)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > 100: 
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            motion_detected = True

    if motion_detected:
        print("Motion detected!")
    else:
        print("No motion detected")

    cv2.imshow('Motion Detection and Response', frame)

    key = cv2.waitKey(30) 
    if key == 27:  
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




