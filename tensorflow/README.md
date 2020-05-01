
# Description
This project provide a **single** tensorflow model implemented the mtcnn face detector.
 It is very handy for face detection in python and easy for deployment with tensorflow.
 The model is converted and modified from the original author's caffe model.
 
 For more detail about mtcnn, see the
  [original project](https://github.com/kpzhang93/MTCNN_face_detection_alignment).

# Requirement
- tensorflow >= 1.5.0 (older version may work as well, but it is not tested)
- opencv python binding (for reading image and show the result)
- pycaffe (for convert model from caffe to tensorflow)

# Run
```bash
# simple detection demo
python mtcnn.py test_image.jpg

# A demo shows how to use tensorflow dataset api
# to accelerate detection with multi-cores. This is
# especially useful for processing large amount of
# small image data in a powerful server.
python mtcnn_data.py imglist.txt result
```

# Convert model
The default model `mtcnn.pb` will work well. But if you want to modify the model's behave, you may need
to convert the model yourself.
```bash
# download model from original project
git clone https://github.com/kpzhang93/MTCNN_face_detection_alignment
# convert model
python caffe2tf.py MTCNN_face_detection_alignment/code/codes/MTCNNv1/model ./mtcnn.pb
```

# Result
![result.jpg](./result.jpg)

# Input and Ouput
## Input: 
 BGR image.
## Output:
- box: bouding box, 2D float tensor with format [[y1, x1, y2, x2], ...]
- prob: confidence, 1D float tensor with format [x, ...]
- landmarks: face landmarks, 2D float tensor with format[[y1, y2, y3, y4, y5, x1, x2, x3, x4, x5], ...]

# Note
- Because the model is designed to work with opencv, so the input image format is BGR instead of RGB. If 
you prefer RGB, you can modify the convert script and convert the model yourself.
- The convert code make the model more suitable for tensorflow and opencv by modifying the model's parameters. 
