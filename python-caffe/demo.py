import cv2,os
import numpy as np
from MtcnnDetector import MTCNNDetector

def test_dir(dir="../imgs"):
    files=os.listdir(dir)
    for file in files:
        imgpath=dir+"/"+file
        img = cv2.imread(imgpath)
        detector = FaceDetector(minsize = 20,fastresize = False) 
        total_boxes,points,numbox = detector.detectface(img)
        for i in range(numbox):
            cv2.rectangle(img,(int(total_boxes[i][0]),int(total_boxes[i][1])),(int(total_boxes[i][2]),int(total_boxes[i][3])),(0,255,0),2)        
            for j in range(5):        
                cv2.circle(img,(int(points[j,i]),int(points[j+5,i])),2,(0,0,255),2)
        cv2.imshow( 'img',img )
        cv2.waitKey()

def test_camera(index=0):
    cap=cv2.VideoCapture(index)
    detector = MTCNNDetector(minsize = 20, fastresize = False) 
    while True:
        ret,img=cap.read()
        if not ret:
            break
        total_boxes,points,numbox = detector.detectface(img)
        for i in range(numbox):
            cv2.rectangle(img,(int(total_boxes[i][0]),int(total_boxes[i][1])),(int(total_boxes[i][2]),int(total_boxes[i][3])),(0,255,0),2)        
            for j in range(5):        
                cv2.circle(img,(int(points[j,i]),int(points[j+5,i])),2,(0,0,255),2)
        cv2.imshow( 'img',img )
        cv2.waitKey(1)

if __name__ == '__main__':
    #test_dir()
    test_camera()