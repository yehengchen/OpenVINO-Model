import cv2

import imutils

import time

model_path = "model/age-gender-recognition-retail-0013.xml"

pbtxt_path = "model/age-gender-recognition-retail-0013.bin"

net = cv2.dnn.readNet(model_path, pbtxt_path)

face_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

cascade_scale = 1.2

cascade_neighbors = 6

minFaceSize = (30,30)

net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

def getFaces(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor= cascade_scale,

        minNeighbors=cascade_neighbors,

        minSize=minFaceSize,

        flags=cv2.CASCADE_SCALE_IMAGE

    )

    bboxes = []

    for (x,y,w,h) in faces:

        if(w>minFaceSize[0] and h>minFaceSize[1]):

            bboxes.append((x, y, w, h))

    return bboxes

camera = cv2.VideoCapture(0)

frameID = 0

grabbed = True

start_time = time.time()

while grabbed:

    (grabbed, img) = camera.read()

    img = cv2.resize(img, (550,400))

    out = []

    frame = img.copy()

    faces = getFaces(frame)

    x, y, w, h = 0, 0, 0, 0

    i = 0

    for (x,y,w,h) in faces:

        cv2.rectangle( frame,(x,y),(x+w,y+h),(0,255,0),2)

        if(w>0 and h>0):

            facearea = frame[y:y+h, x:x+w]

            blob = cv2.dnn.blobFromImage(facearea, size=(62, 62), ddepth=cv2.CV_8U)

            net.setInput(blob)

            out = net.forward()

            num_age = out[0][0][0][0]

            num_sex = out[0][1][0][0]

            age = int(num_age*100)

            if(num_sex>0.5):

                sex = "man"

            else:

                sex = "woman"

            txt = "sex: {}, age: {}".format(sex,age)

            if(age<=1):

                txt = "sex: {}, age: {}".format(sex,'?')

            if(i % 2 == 0):

                cv2.putText(frame,txt,(int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 255, 0), 2)

            else:

                cv2.putText(frame,txt,(int(x), int(y+h)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255, 255, 0), 2)

            i += 1

    cv2.imshow("FRAME", frame)

    frameID += 1

    fps = frameID / (time.time() - start_time)

    print("FPS:", fps)

    cv2.waitKey(1)
