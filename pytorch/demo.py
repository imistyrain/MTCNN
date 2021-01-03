import cv2
import time
import argparse
import math
import numpy as np
import torch
from MTCNN_nets import PNet, RNet, ONet
from utils.util import *

def get_args():
    parser = argparse.ArgumentParser(description='MTCNN Demo')
    parser.add_argument("--test_image", default="images/test.jpg")
    parser.add_argument('--mini_face', default="80", type=int)
    parser.add_argument('--camera', default=True)
    args = parser.parse_args()
    return args

def create_mtcnn_net(image, mini_face, device, p_model_path=None, r_model_path=None, o_model_path=None):

    boxes = np.array([])
    landmarks = np.array([])

    if p_model_path is not None:
        pnet = PNet().to(device)
        pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
        pnet.eval()

        bboxes = detect_pnet(pnet, image, mini_face, device)


    if r_model_path is not None:
        rnet = RNet().to(device)
        rnet.load_state_dict(torch.load(r_model_path, map_location=lambda storage, loc: storage))
        rnet.eval()

        bboxes = detect_rnet(rnet, image, bboxes, device)

    if o_model_path is not None:
        onet = ONet().to(device)
        onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        onet.eval()

        bboxes, landmarks = detect_onet(onet, image, bboxes, device)

    return bboxes, landmarks
 
def detect_pnet(pnet, image, min_face_size, device):

    # start = time.time()

    thresholds = 0.6 # face detection thresholds
    nms_thresholds = 0.7

    # BUILD AN IMAGE PYRAMID
    height, width, channel = image.shape
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that minimum size that we can detect equals to minimum face size that we want to detect
    m = min_detection_size / min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor ** factor_count)
        min_length *= factor
        factor_count += 1

    # it will be returned
    bounding_boxes = []

    with torch.no_grad():
        # run P-Net on different scales
        for scale in scales:
            sw, sh = math.ceil(width * scale), math.ceil(height * scale)
            img = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
            img = torch.FloatTensor(preprocess(img)).to(device)
            offset, prob = pnet(img)
            probs = prob.cpu().data.numpy()[0, 1, :, :]  # probs: probability of a face at each sliding window
            offsets = offset.cpu().data.numpy()  # offsets: transformations to true bounding boxes
            # applying P-Net is equivalent, in some sense, to moving 12x12 window with stride 2
            stride, cell_size = 2, 12
            # indices of boxes where there is probably a face
            # returns a tuple with an array of row idx's, and an array of col idx's:
            inds = np.where(probs > thresholds)

            if inds[0].size == 0:
                boxes = None
            else:
                # transformations of bounding boxes
                tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
                offsets = np.array([tx1, ty1, tx2, ty2])
                score = probs[inds[0], inds[1]]
                # P-Net is applied to scaled images
                # so we need to rescale bounding boxes back
                bounding_box = np.vstack([
                    np.round((stride * inds[1] + 1.0) / scale),
                    np.round((stride * inds[0] + 1.0) / scale),
                    np.round((stride * inds[1] + 1.0 + cell_size) / scale),
                    np.round((stride * inds[0] + 1.0 + cell_size) / scale),
                    score, offsets])
                boxes = bounding_box.T
                keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
                boxes[keep]

            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds)
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bboxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5],  x1, y1, x2, y2, score

        bboxes = convert_to_square(bboxes)
        bboxes[:, 0:4] = np.round(bboxes[:, 0:4])

        # print("pnet predicted in {:2.3f} seconds".format(time.time() - start))

        return bboxes

def detect_rnet(rnet, image, bboxes, device):

    # start = time.time()

    size = 24
    thresholds = 0.7  # face detection thresholds
    nms_thresholds = 0.7
    height, width, channel = image.shape

    num_boxes = len(bboxes)
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)

    img_boxes = np.zeros((num_boxes, 3, size, size))

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3))

        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
            image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = cv2.resize(img_box, (size, size), interpolation=cv2.INTER_LINEAR)

        img_boxes[i, :, :, :] = preprocess(img_box)

    img_boxes = torch.FloatTensor(img_boxes).to(device)
    offset, prob = rnet(img_boxes)
    offsets = offset.cpu().data.numpy()  # shape [n_boxes, 4]
    probs = prob.cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds)[0]
    bboxes = bboxes[keep]
    bboxes[:, 4] = probs[keep, 1].reshape((-1,))  # assign score from stage 2
    offsets = offsets[keep]  #

    keep = nms(bboxes, nms_thresholds)
    bboxes = bboxes[keep]
    bboxes = calibrate_box(bboxes, offsets[keep])
    bboxes = convert_to_square(bboxes)
    bboxes[:, 0:4] = np.round(bboxes[:, 0:4])

    # print("rnet predicted in {:2.3f} seconds".format(time.time() - start))

    return bboxes

def detect_onet(onet, image, bboxes, device):

    # start = time.time()

    size = 48
    thresholds = 0.8  # face detection thresholds
    nms_thresholds = 0.7
    height, width, channel = image.shape

    num_boxes = len(bboxes)
    [dy, edy, dx, edx, y, ey, x, ex, w, h] = correct_bboxes(bboxes, width, height)

    img_boxes = np.zeros((num_boxes, 3, size, size))

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3))

        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = \
            image[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = cv2.resize(img_box, (size, size), interpolation=cv2.INTER_LINEAR)

        img_boxes[i, :, :, :] = preprocess(img_box)

    img_boxes = torch.FloatTensor(img_boxes).to(device)
    landmark, offset, prob = onet(img_boxes)
    landmarks = landmark.cpu().data.numpy()  # shape [n_boxes, 10]
    offsets = offset.cpu().data.numpy()  # shape [n_boxes, 4]
    probs = prob.cpu().data.numpy()  # shape [n_boxes, 2]

    keep = np.where(probs[:, 1] > thresholds)[0]
    bboxes = bboxes[keep]
    bboxes[:, 4] = probs[keep, 1].reshape((-1,))  # assign score from stage 2
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bboxes[:, 2] - bboxes[:, 0] + 1.0
    height = bboxes[:, 3] - bboxes[:, 1] + 1.0
    xmin, ymin = bboxes[:, 0], bboxes[:, 1]

    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

    bboxes = calibrate_box(bboxes, offsets)
    keep = nms(bboxes, nms_thresholds, mode='min')
    bboxes = bboxes[keep]
    landmarks = landmarks[keep]

    # print("onet predicted in {:2.3f} seconds".format(time.time() - start))

    return bboxes, landmarks

def show(image, bboxes, landmarks):
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, :4]
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            landmark = landmarks[i, :]
            landmark = landmark.reshape(2, 5).T
            for j in range(5):
                cv2.circle(image, (int(landmark[j, 0]), int(landmark[j, 1])), 2, (0, 255, 255), 1)
    return image

if __name__ == '__main__':
    args = get_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.camera:
        cap = cv2.VideoCapture(0)
        while True:
            ret, image = cap.read()
            if not ret:
                break
            start = time.time()
            bboxes, landmarks = create_mtcnn_net(image, args.mini_face, device, p_model_path='model/pytorch/pnet_Weights', r_model_path='model/pytorch/rnet_Weights', o_model_path='model/pytorch/onet_Weights')
            print("image predicted in {:2.3f} seconds".format(time.time() - start))
            image = show(image, bboxes, landmarks) 
            cv2.imshow('image', image)
            cv2.waitKey(1)
    else:
        image = cv2.imread(args.test_image)
        bboxes, landmarks = create_mtcnn_net(image, args.mini_face, device, p_model_path='model/pytorch/pnet_Weights', r_model_path='model/pytorch/rnet_Weights', o_model_path='model/pytorch/onet_Weights')
        image = show(image, bboxes, landmarks) 
        cv2.imshow('image', image)
        cv2.waitKey()