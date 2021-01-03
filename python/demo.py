import os
import cv2
import math
import numpy as np

try:
    # use caffe for inference, build pycaffe
    from MtcnnDetector import MTCNNDetector
    print("use caffe for inference")
except:
    # use opencv for inference
    # pip install opencv-python==3.4.8.29
    print("pycaffe cann't be imported, use opencv for inferce")
    from Mtcnndnn import MTCNNDetector

colors = [[255,0,0],[0,255,0],[0,0,255]]
objectPoints = np.array([[2.37427, 110.322, 21.7776],
                         [70.0602, 109.898, 20.8234],
                         [36.8301, 78.3185, 52.0345],
                         [14.8498, 51.0115, 30.2378],
                         [58.1825, 51.0115, 29.6224]])

def drawDetection(img, boxes, points):
    for i in range(boxes.shape[0]):
        box = boxes[i]
        box = [ int(b) for b in box]
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,255,0),2)        
        for j in range(5):        
            cv2.circle(img,(int(points[j,i]),int(points[j+5,i])),2,(0,0,255), -1)
            
def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    x = x*180.0/3.141592653589793
    y = y*180.0/3.141592653589793
    z = z*180.0/3.141592653589793
    return x,y,z

def estimateHeadPose(img, points):
    if len(points) == 0 or points.shape[1] == 0:
        return
    height, width, _ = img.shape
    focal_length = (width+height)/2
    camera_matrix = np.zeros([3,3], dtype = np.float32)
    camera_matrix[0,:] = [focal_length, 0, width/2]
    camera_matrix[1,:] = [ 0, focal_length, height/2]
    camera_matrix[2,:] = [0, 0, 1]
    dist_coeffs = np.zeros([5,1], np.float32)
    imagePoints = []
    for i in range(5):
        imagePoints.append([points[i],points[i+5]])
    imagePoints = np.array(imagePoints,dtype=np.float32).reshape(-1,2)
    ret, rVec, tVec = cv2.solvePnP(objectPoints, imagePoints, camera_matrix, dist_coeffs, flags = cv2.SOLVEPNP_EPNP)
    if ret:
        # draw 3d axis
        axislength = 40
        mp2 = objectPoints[2]
        axis3d = np.array([[mp2[0],mp2[1],mp2[2]],
                        [mp2[0]+axislength,mp2[1],mp2[2]],
                        [mp2[0],mp2[1]+axislength,mp2[2]],
                        [mp2[0],mp2[1],mp2[2]+axislength]],dtype = np.float32)
        axis2d, _ = cv2.projectPoints(axis3d,rVec,tVec,camera_matrix, dist_coeffs)
        axis2d = np.int32(axis2d).reshape(-1,2)
        pt0 = axis2d[0]
        for i in range(3):
            ptaxis = axis2d[i+1]
            cv2.line(img, tuple(pt0),tuple(ptaxis),colors[i],3)

        R = cv2.Rodrigues(rVec)[0]
        T = np.hstack((R,tVec))
        roll, pitch, yaw = cv2.decomposeProjectionMatrix(T, camera_matrix)[-1]
        eulers= [roll, pitch, yaw]
        eulers2 = rotationMatrixToEulerAngles(R)
        print(eulers2)
        for i in range(3):
            cv2.putText(img, str(eulers[i]),(0,40+20*i),3,1,colors[i])
        return eulers

def test_dir(detector, dir = "images"):
    files = os.listdir(dir)
    for file in files:
        imgpath = dir + "/" + file
        img = cv2.imread(imgpath)
        if img is None:
            continue
        boxes, points = detector.detect(img)
        estimateHeadPose(img, points)
        drawDetection(img, boxes, points)
        cv2.imshow('img',img)
        cv2.waitKey()

def test_camera(detector):
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        boxes, points= detector.detect(img)
        estimateHeadPose(img, points)
        drawDetection(img, boxes, points)
        cv2.imshow('img',img)
        cv2.waitKey(1)

if __name__ == '__main__':
    detector = MTCNNDetector() 
    test_dir(detector)
    #test_camera(detector)