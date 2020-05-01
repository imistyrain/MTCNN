# MTCNN

## 概述

[MTCNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)是[Kaipeng Zhang](https://kpzhang93.github.io/)等人提出的多任务级联卷积神经网络进行人脸检测的方法，是迄今为止开放源码的效果最好的人脸检测器之一，[在fddb上有100个误报时的检出率高达90%以上](https://github.com/imistyrain/fddb-windows)，作者提供的版本为[matlab版](https://github.com/kpzhang93/MTCNN_face_detection_alignment),它采用三级级联架构分阶段逐步过滤人脸，在CPU上可达到实时和较高的准确率，是目前人脸检测领域的baseline：

![](https://kpzhang93.github.io/MTCNN_face_detection_alignment/support/index.png)

本项目集成了其在OpenCV dnn、caffe、ncnn和tensorflow等框架的使用，可以跨Windows、Linux和Mac运行.

它最终的效果如图所示：

![](https://i.imgur.com/FbglxoX.jpg)

## 使用方法

1.按照[MRHead](https://github.com/imistyrain/MRHead)描述的方法配置好opencv跨平台编译环境

2.编译最新版[caffe](https://github.com/BVLC/caffe)，这个网上已有很多[教程](http://blog.csdn.net/akashaicrecorder/article/details/71016942),恕不赘述
```
git clone https://github.com/BVLC/caffe
cd caffe
git checkout windows
script\build_win.cmd
```

3.打开MTCNN.sln，把mtcnn设为启动项

4.设置所需的环境变量

## 参考

*  [Win10+VS2015 caffe环境搭建](http://blog.csdn.net/akashaicrecorder/article/details/71016942) 可选用跨平台版本[ssd](https://github.com/imistyrain/ssd)

* [MTCNN-light](https://github.com/AlphaQi/MTCNN-light)

* 训练[Joint-Face-Detection-and-Alignment](https://github.com/luoyetx/Joint-Face-Detection-and-Alignment)

* [MTCNN训练统计分析](https://blog.csdn.net/minstyrain/article/details/83089196)