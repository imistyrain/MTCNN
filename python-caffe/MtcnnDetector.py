import os
import sys
import cv2
import platform
import math
import numpy as np
if platform.system()=="Windows":
    caffe_root = 'D:/CNN/ssd'
else:
    caffe_root = os.path.expanduser('~') + "/CNN/ssd"
sys.path.insert(0, caffe_root + '/python')
import caffe

class MTCNNDetector(object):
    '''
        Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks
        see https://github.com/kpzhang93/MTCNN_face_detection_alignment
        this is a caffe version
    '''
    def __init__(self,
                 minsize = 20,
                 threshold = [0.6, 0.7, 0.7],
                 factor = 0.709,
                 fastresize = False,
                 model_dir="../model/caffe",
                 gpuid = 0):
        
        self.minsize = minsize
        self.threshold = threshold
        self.factor = factor
        self.fastresize = fastresize
        
        model_P = model_dir+'/det1.prototxt'
        weights_P = model_dir+'/det1.caffemodel'
        model_R = model_dir+'/det2.prototxt'
        weights_R = model_dir+'/det2.caffemodel'
        model_O = model_dir+'/det3.prototxt'
        weights_O = model_dir+'/det3.caffemodel'
        
        caffe.set_mode_gpu()
        caffe.set_device(gpuid)
        
        self.PNet = caffe.Net(model_P, weights_P, caffe.TEST) 
        self.RNet = caffe.Net(model_R, weights_R, caffe.TEST)
        self.ONet = caffe.Net(model_O, weights_O, caffe.TEST)     
        

    def bbreg(self,boundingbox,reg):
    
        '''Calibrate bounding boxes'''
        
        if reg.shape[1]==1:
            reg = np.shape(reg,(reg.shape[2],reg.shape[3])).T
        w = boundingbox[:,2]-boundingbox[:,0]+1
        h = boundingbox[:,3]-boundingbox[:,1]+1
        boundingbox[:,0:4] = np.reshape(np.hstack((boundingbox[:,0]+reg[:,0]*w, boundingbox[:,1]+reg[:,1]*h, boundingbox[:,2]+reg[:,2]*w, boundingbox[:,3]+reg[:,3]*h)),(4,w.shape[0])).T
    
        return boundingbox
    
    def nms(self,dets, thresh,type='Union'):
        
        if dets.shape[0]==0:
            keep = []
            return keep

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
    
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
    
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
    
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if type=='Min':
                ovr = inter / np.minimum(areas[i], areas[order[1:]])  
            else:
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
    
        return keep
        
    def rerec(self,bboxA):
        
        '''Convert bboxA to square'''
        
        h = bboxA[:,3]-bboxA[:,1]
        w = bboxA[:,2]-bboxA[:,0]
        l = np.concatenate((w,h)).reshape((2,h.shape[0]))
        l = np.amax(l, axis=0) 
        bboxA[:,0] = bboxA[:,0] + w*0.5 -l*0.5
        bboxA[:,1] = bboxA[:,1] + h*0.5 -l*0.5
        bboxA[:,2] = bboxA[:,0] + l
        bboxA[:,3] = bboxA[:,1] + l
    
        return bboxA
    
    def sort_rows_by_icol1(self,inarray):

        idex=np.lexsort([inarray[:,0],inarray[:,1]])
        a_sort=inarray[idex,:]
        return a_sort
    
    
    def generateBoundingBox(self,map,reg,scale,threshold):
    
        '''Use heatmap to generate bounding boxes'''
        
        stride=2;
        cellsize=12;
        boundingbox=[];
        
        map = map.T
        dx1=reg[:,:,0].T
        dy1=reg[:,:,1].T
        dx2=reg[:,:,2].T
        dy2=reg[:,:,3].T
  
        [y,x]=np.where(map>=threshold)
        y = np.reshape(y,(len(y),1))
        x = np.reshape(x,(len(y),1))
        a = np.where(map.flatten(1)>=threshold)

        if y.shape[0]==1:
            y=y.T
            x=x.T
            score=np.reshape(map.flatten(1)[a[0]],(1,1))
            dx1=dx1.T
            dy1=dy1.T
            dx2=dx2.T
            dy2=dy2.T
        else:

            score=map.flatten(1)[a[0]]
            score=np.reshape(score, (a[0].shape[0],1))
            
        dx1N=np.reshape(dx1.flatten(1)[a[0]], (a[0].shape[0],1))
        dy1N=np.reshape(dy1.flatten(1)[a[0]], (a[0].shape[0],1))
        dx2N=np.reshape(dx2.flatten(1)[a[0]], (a[0].shape[0],1))
        dy2N=np.reshape(dy2.flatten(1)[a[0]], (a[0].shape[0],1))  
        
        reg=np.hstack((dx1N,dy1N,dx2N,dy2N))
        
        if  reg.shape[0]==0:
            reg = np.zeros(shape=(0,3))
        
        boundingbox=np.hstack((y,x))
        boundingbox = self.sort_rows_by_icol1(boundingbox)
        boundingbox=np.hstack((((stride*boundingbox+1)/scale-1).astype(int),(((stride*boundingbox+cellsize-1+1)/scale-1)).astype(int),score,reg))

        return boundingbox
    
    def pad(self,total_boxes,w,h):
    
        '''Compute the padding coordinates (pad the bounding boxes to square)'''
        
        tmpw=total_boxes[:,2]-total_boxes[:,0]+1
        tmph=total_boxes[:,3]-total_boxes[:,1]+1
        numbox=total_boxes.shape[0]
        
        dx = np.ones((numbox,))
        dy = np.ones((numbox,))
        
        edx = tmpw    
        edy = tmph
            
        x = total_boxes[:,0]
        y = total_boxes[:,1]
        ex = total_boxes[:,2]
        ey = total_boxes[:,3]
        
        tmp = np.where(ex>w)
        edx[tmp] = -ex[tmp] + w + tmpw[tmp]
        ex[tmp] = w
        
        tmp = np.where(ey>h)
        edy[tmp]= -ey[tmp] + h + tmph[tmp]
        ey[tmp] = h
        
        tmp = np.where(x < 1)
        dx[tmp] = 2-x[tmp]
        x[tmp] = 1	
        
        tmp = np.where(y < 1)
        dy[tmp] = 2-y[tmp]
        y[tmp] = 1
        
        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph
    
        
    def LoadNet(self,model,weights):
        caffe.set_mode_gpu()
        caffe.set_device(0)
        Net = caffe.Net(model, weights, caffe.TEST)
        return Net
    
    def detectface(self,img):

        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        if self.fastresize:
            im_data=(np.float32(img)-127.5)*0.0078125
        
        
        factor_count=0
        total_boxes=[]
        points=[]
        h=img.shape[0]
        w=img.shape[1]

        minl=min(w,h)
        m=12.0/self.minsize
        minl=minl*m
        # creat scale pyramid
        scales=[]
        while (minl>=12.0):
            scales.append(m*(math.pow(self.factor,factor_count)))
            minl=minl*self.factor
            factor_count=factor_count+1

        total_boxes = np.zeros(shape=(0,9))

        for scale in scales:
        
            hs=int(math.ceil(h*scale))
            ws=int(math.ceil(w*scale))
            if self.fastresize:
                im_data_out = cv2.resize(im_data,(ws, hs),interpolation=cv2.INTER_NEAREST)

            else:
                im_data_out = (cv2.resize(img,(ws, hs),interpolation=cv2.INTER_NEAREST) - 127.5)*0.0078125
            im_data_out = im_data_out[None,:] 
            im_data_out = im_data_out.transpose((0,3,2,1)) 
            self.PNet.blobs['data'].reshape(1,3,ws,hs)
            out = self.PNet.forward_all( data = im_data_out )
            
            
            map = out['prob1'][0].transpose((2,1,0))[:,:,1]
            reg = out['conv4-2'][0].transpose((2,1,0))
            boxes = self.generateBoundingBox(map,reg,scale,self.threshold[0])
            
            pick = self.nms(boxes, 0.5)
            boxes = boxes[pick,:]
            if boxes.shape[0]!=0:
                total_boxes = np.concatenate((total_boxes,boxes),axis=0)
        

        if total_boxes is not None:
            pick = self.nms(total_boxes, 0.7)
            total_boxes = total_boxes[pick,:]
            regw=total_boxes[:,2]-total_boxes[:,0];
            regh=total_boxes[:,3]-total_boxes[:,1];
            total_boxes = np.concatenate((total_boxes[:,0]+total_boxes[:,5]*regw, total_boxes[:,1]+total_boxes[:,6]*regh, total_boxes[:,2]+total_boxes[:,7]*regw, total_boxes[:,3]+total_boxes[:,8]*regh, total_boxes[:,4])).reshape((5,regw.shape[0]))   
            total_boxes = total_boxes.T
            total_boxes=self.rerec(total_boxes)
            total_boxes[:,0:4]=total_boxes[:,0:4].astype(int)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes,w,h)
            
        numbox = total_boxes.shape[0]

        
        if  numbox > 0:    
            #second stage
            tempimg =  np.zeros((24,24,3,numbox))
            for k in range(numbox):
                tmp =  np.zeros(((int)(tmph[k]),(int)(tmpw[k]),3))
                tmp[(int)(dy[k])-1:(int)(edy[k]),(int)(dx[k])-1:(int)(edx[k]),:]=img[(int)(y[k])-1:(int)(ey[k]),(int)(x[k])-1:(int)(ex[k]),:] 
                tempimg[:,:,:,k]= cv2.resize(tmp,(24, 24),interpolation=cv2.INTER_NEAREST)
            tempimg = (tempimg-127.5)*0.0078125
            tempimg = tempimg.transpose((3,2,1,0)) 
            self.RNet.blobs['data'].reshape(numbox,3,24,24)
            out = self.RNet.forward_all( data = tempimg )        

            score=out['prob1'][:,1]   ###why need to squeeze?
            pas = np.where(score>self.threshold[1])            
            total_boxes = np.hstack((total_boxes[pas[0],0:4], np.reshape(score[pas[0]],(len(pas[0]),1))))
            mv = out['conv5-2'][pas[0],:]

            if total_boxes is not None:
                pick = self.nms(total_boxes, 0.7)
                total_boxes = total_boxes[pick,:]  
                total_boxes=self.bbreg(total_boxes, mv[pick,:])
                total_boxes=self.rerec(total_boxes)
                
            numbox = total_boxes.shape[0]
        
            if  numbox > 0: 
                # third stage
                total_boxes = total_boxes.astype(int)
                dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes,w,h)
                tempimg =  np.zeros((48,48,3,numbox))
                for k in range(numbox):
                    tmp =  np.zeros((tmph[k],tmpw[k],3))
                    tmp[(int)(dy[k])-1:(int)(edy[k]),(int)(dx[k])-1:(int)(edx[k]),:]=img[y[k]-1:ey[k],x[k]-1:ex[k],:] 
                    tempimg[:,:,:,k]= cv2.resize(tmp,(48, 48),interpolation=cv2.INTER_NEAREST)    
                tempimg = (tempimg-127.5)*0.0078125 
                tempimg = tempimg.transpose((3,2,1,0)) 
                self.ONet.blobs['data'].reshape(numbox,3,48,48)
                out = self.ONet.forward_all( data = tempimg ) 
        
                score = out['prob1'][:,1]
                points = out['conv6-3']
                pas = np.where(score>self.threshold[2])
                points = points[pas[0],:].T
                total_boxes = np.hstack((total_boxes[pas[0],0:4], np.reshape(score[pas[0]],(len(pas[0]),1))))
                mv = out['conv6-2'][pas[0],:]
                w=total_boxes[:,2]-total_boxes[:,0]+1
                h=total_boxes[:,3]-total_boxes[:,1]+1
                points[0:5,:] = np.tile(np.reshape(w,(1,w.shape[0])),[5,1])*points[0:5,:]+np.tile(np.reshape(total_boxes[:,0],(1,total_boxes.shape[0])),[5,1])-1
                points[5:10,:] = np.tile(np.reshape(h,(1,h.shape[0])),[5,1])*points[5:10,:]+np.tile(np.reshape(total_boxes[:,1],(1,total_boxes.shape[0])),[5,1])-1
                if total_boxes is not None:
                    total_boxes=self.bbreg(total_boxes, mv[:,:])
                    pick = self.nms(total_boxes, 0.7, 'Min')
                    total_boxes = total_boxes[pick,:]
                    points = points[:,pick]
            numbox = total_boxes.shape[0]       
        return total_boxes,points,numbox
    

