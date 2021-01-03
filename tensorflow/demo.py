import argparse,cv2
from mtcnn import MTCNN

def test_image(imgpath):
    mtcnn = MTCNN('./mtcnn.pb')
    img = cv2.imread(imgpath)
    show=mtcnn.detectAndDraw(img)
    cv2.imshow('img', show)
    cv2.waitKey()

def test_camera(index=0):
    mtcnn = MTCNN('./mtcnn.pb')
    cap=cv2.VideoCapture(index)
    while True:
        ret,img=cap.read()
        if not ret:
            break
        show=mtcnn.detectAndDraw(img)
        cv2.imshow('img', show)
        cv2.waitKey(1)

if __name__ == '__main__':
    #test_image()
    test_camera()