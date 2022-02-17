#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import time 
import numpy as np
import imutils
import datetime
import os


# In[5]:


def gen():
    """Video streaming generator function."""
    gun_cascade = cv2.CascadeClassifier('cascade.xml')
    #cap = cv2.VideoCapture(0)
    firstFrame = None

# loop over the frames of the video

    gun_exist = False
    # Read until video is completed
    while True:
        # (grabbed, frame) = cap.read()
        frame = cap.read()
        # if the frame could not be grabbed, then we have reached the end of the video
        # if not grabbed:
        #     break

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize = (100, 100))
        print(len(gun))
        if len(gun) > 0:
            gun_exist = True
            
        for (x,y,w,h) in gun:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]    

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
            continue

        # draw the text and timestamp on the frame
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


        if gun_exist:
            print("guns detected")
            cv2.imwrite("detected.jpg", frame)
            # sendMail()
        else:
            print("guns NOT detected")
        gun_exist = False
        # show the frame and record if the user presses a key
        
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        
   


# In[19]:


images = []


# In[23]:



def fun(folder):
    for filenam in os.listdir(folder):
        tup = (filenam,1)
        images.append(tup)


# In[24]:


fun(r'C:\Users\Pranav\OneDrive\Desktop\New folder\gun-data\1')


# In[31]:


import random
random.shuffle(images)


# In[32]:


print(images[3000:4000])


# In[35]:


images[0][0]


# In[86]:


def findAccuracy(images):
    
    gun_cascade = cv2.CascadeClassifier('cascade.xml')
#     firstFrame = gray

# loop over the frames of the video

    gun_exist = False
    count1 = 0
    count2 = 0
    for file in images[:250]:
        
        frame = cv2.imread(os.path.join(r'C:\Users\Pranav\OneDrive\Desktop\New folder\gun-data\combine',file[0]))
        # if the frame could not be grabbed, then we have reached the end of the video
        # if not grabbed:
        #     break

        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gun = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize = (100, 100))
#         print(len(gun))
        if len(gun) > 0:
            gun_exist = True
            
        for (x,y,w,h) in gun:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]    

#         # if the first frame is None, initialize it
#         if firstFrame is None:
#             firstFrame = gray
#             continue

        # draw the text and timestamp on the frame
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        print(file[0], file[1], gun_exist)
        if gun_exist:
            if file[1]==1:
                count1 = count1 + 1
            else:
                count2 = count2 + 1
                
#             print("guns detected")
#             cv2.imwrite("detected.jpg", frame)
            # sendMail()
        else:
            if file[1]==1:
                count2 = count2 + 1
            else:
                count1 = count1 + 1
#             print("guns NOT detected")
        gun_exist = False
    
    print(count1, count2)
    return count1,count2
        # show the frame and record if the user presses a key


# In[87]:


count1,count2 = findAccuracy(images)


# In[88]:


count1/(count1+count2)


# In[69]:


print(len(images))


# In[73]:


import keras
model = keras.models.load_model('pre-model2.h5')


# In[83]:


def tensorModel(images):
    right,wrong=0,0
    counter = 0
    for file in images[:250]:
        img = cv2.imread(os.path.join(r'C:\Users\Pranav\OneDrive\Desktop\New folder\gun-data\combine',file[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (416,416) )
        img = np.array(img)
        y_hat = model.predict(img.reshape(-1,416,416,3))[0]
        if y_hat[1]>0.40:
            if file[1]==1:
                right+=1
            else:
                wrong+=1
        else:
            if file[1]==1:
                wrong+=1
            else:
                right+=1
        counter+=1

    return right,wrong


# In[84]:


right,wrong = tensorModel(images)
print(right, wrong)


# In[85]:


print(right/(wrong+right))


# In[ ]:




