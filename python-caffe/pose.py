#coding=utf-8
import cv2,os,sys
import numpy as np
from MtcnnDetector import MTCNNDetector
import math

caffe_root = os.path.expanduser('~') + "/CNN/ssd"
sys.path.insert(0, caffe_root+'/python')
import caffe
input_width = 64
input_height = 64

def preprocess(im, bbox):
    pad = cal_padding(bbox, im)
    im_pad = cv2.copyMakeBorder(im, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    bbox = bbox + pad
    bb_w = bbox[2] - bbox[0]
    scale = bb_w * 1.0 / input_width
    h, w, c = im_pad.shape
    bbox = bbox / scale
    bbox[0] = round(bbox[0])
    bbox[1] = round(bbox[1])
    bbox[2:] = bbox[0:2] + [input_width - 1, input_height - 1]
    bbox = bbox.astype(np.int32)
    im_pad = cv2.resize(im_pad, (int(w / scale), int(h / scale)))
    cropImg = im_pad[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
    return cropImg,pad,bbox[0:2]+1,scale

def cal_padding(bbox, im):
    x1,y1,x2,y2 = bbox
    h,w,c = im.shape
    pad = np.max([-x1, -y1, x2 - w, y2 - h, 0]) + 10
    return int(pad)

def pad_bbox(bbox, pad_ratio):
    # padding
    pad_w = (bbox[2] - bbox[0]) * pad_ratio
    pad_h = (bbox[3] - bbox[1]) * pad_ratio
    bbox = np.array([bbox[0] - pad_w, bbox[1], bbox[2] + pad_w, bbox[3] + 2 * pad_h])
    return np.array(bbox)

def obtain_bbox(bbox, w_h_ratio):

    bbox = np.array(bbox).astype(np.float32)

    w, h = bbox[2:] - bbox[0:2] + 1
    if w*1.0/h >= w_h_ratio:
        pad_h = (w/w_h_ratio -h)/2
        pad_w = 0
    elif w/h < w_h_ratio:
        pad_h = 0
        pad_w = (h*w_h_ratio -w)/2
    bbox = bbox[0] - pad_w, bbox[1] - pad_h, bbox[2] + pad_w, bbox[3] + pad_h
    return np.array(bbox)

def rotMatrixToEulerAngle(rotMat):
    theta = cv2.norm(rotMat,cv2.NORM_L2)
    w = np.cos(theta/2);
    x = np.sin(theta/2) * rotMat[0] / theta
    y = np.sin(theta/2) * rotMat[1] / theta
    z = np.sin(theta/2) * rotMat[2] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x * y * z)
    t1 = 1.0 - 2.0 *(x * x + ysqr)
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    elif t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    return roll, yaw, pitch

class PoseEstimator:
    def __init__(self,model_dir="model"):
        model_def = model_dir+'/deploy.prototxt'
        model_weights= model_dir+'/mobilenet-v2.caffemodel'
        self.net=caffe.Net(model_def,model_weights, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))  # h,w,c-> c,h,w
        self.detector = MTCNNDetector(minsize = 20,fastresize = False)

    def predict(self,img):
        total_boxes,points,numbox = self.detector.detectface(img)
        if numbox==0:
            return [],[]
        bbox=total_boxes[0][0:4]
        pt5=points[0]        
        bbox = np.array(bbox, dtype=np.float32)
        bbox = obtain_bbox(bbox, input_width*1.0/input_height)
        bbox = pad_bbox(bbox, pad_ratio=0.05)
        cropImg, pad, offset, scale = preprocess(img, bbox)
        cropImg = (cropImg - 127.5)/128
        cropImg = self.transformer.preprocess('data', cropImg)
        self.net.blobs['data'].data[...] = cropImg
        out = self.net.forward()['fc4']
        landmarks = out.reshape([2, 68])
        landmarks = np.transpose(landmarks)
        landmarks[:, 0] = landmarks[:, 0] * input_width - 1
        landmarks[:, 1] = landmarks[:, 1] * input_height - 1
        focal_length = input_width
        center = [input_width/2, input_height/2]
        camera_matrix = np.zeros([3,3], dtype=np.double)
        camera_matrix[0,:] = [focal_length, 0, center[0]]
        camera_matrix[1,:] = [ 0, focal_length, center[1]]
        camera_matrix[2,:] = [0, 0, 1]
        dist_coeffs = np.zeros([5,1], np.double)
        objectPoints = np.zeros([6,3,1], dtype=np.double)
        objectPoints[0,:,0] = [0,0,0]
        objectPoints[1,:,0] = [0,-330,-65]
        objectPoints[2,:,0] = [-225,170,-135]
        objectPoints[3,:,0] = [225,170,-135]
        objectPoints[4,:,0] = [-150,-150,-125]
        objectPoints[5,:,0] = [150,-150,-125]
        imagePoints = np.zeros([6,2])
        imagePoints = landmarks[[30, 8, 36, 45, 48, 54],:]
        ret, rotVects, transVects = cv2.solvePnP(objectPoints, imagePoints, camera_matrix, dist_coeffs)
        roll, yaw, pitch = rotMatrixToEulerAngle(rotVects)
        for j in range((int)(len(landmarks)/2)):
            landmarks[j]=(landmarks[j]+ offset ) * scale - pad - 1
        print('roll:%f,yaw:%f,pitch:%f' %(roll,yaw,pitch))
        return bbox,landmarks

    def drawPose(self,img,bbox,landmarks):
        h,w,c,=img.shape
        if len(bbox)>0:
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,255,0))
        for j in range((int)(len(landmarks)/2)):
            cv2.circle(img,((int)(landmarks[j][0]),(int)(landmarks[j][1])),2,(0,0,255),2)


def test_dir(dir="../imgs"):
    ps=PoseEstimator()
    files=os.listdir(dir)
    for file in files:
        imgpath=dir+"/"+file
        img = cv2.imread(imgpath)
        bbox,landmarks=ps.predict(img)
        ps.drawPose(img,bbox,landmarks)
        cv2.imshow('img',img)
        cv2.waitKey()

def test_camera():
    cap=cv2.VideoCapture(0)
    ps=PoseEstimator()
    while True:
        ret,img=cap.read()
        if not ret:
            break
        bbox,landmarks=ps.predict(img)
        ps.drawPose(img,bbox,landmarks)
        cv2.imshow( 'img',img )
        cv2.waitKey(1)

if __name__ == '__main__':   
    #test_dir()
    test_camera()