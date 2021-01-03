import cv2
import numpy as np
import torch
import time
import argparse
from PIL import Image, ImageDraw, ImageFont
from utils.util import *
from MTCNN import create_mtcnn_net

def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MTCNN Video')
    parser.add_argument("--scale", dest='scale', help=
    "input frame scale to accurate the speed", default="0.1", type=float)
    parser.add_argument('--mini_face', dest='mini_face', help=
    "Minimum face to be detected. derease to increase accuracy. Increase to increase speed",
                        default="32", type=int)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(0)
    while True:
        isSuccess, frame = cap.read()
        if isSuccess:
            try:
                start_time = time.time()
                input = resize_image(frame, args.scale)
                bboxes, landmarks = create_mtcnn_net(input, args.mini_face, device, p_model_path='weights/pnet_Weights', r_model_path='weights/rnet_Weights', o_model_path='weights/onet_Weights')

                if bboxes != []:
                    bboxes = bboxes / args.scale
                    landmarks = landmarks / args.scale
                if bboxes != []:
                    bboxes = bboxes.astype(np.int)
                    landmarks = landmarks.astype(np.int)
                    FPS = 1.0 / (time.time() - start_time)
                    cv2.putText(frame,'FPS: {:.1f}'.format(FPS),(10,50),3,1,(0,0,255))
                    for i, b in enumerate(bboxes):
                        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (255,0,0),3)
                    for p in landmarks:
                        for i in range(5):
                            cv2.circle(frame,(p[i],p[i+5]),3,(0,255,0))
                
            except Exception as e:
                print(e)

            cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()