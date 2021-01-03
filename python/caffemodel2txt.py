#coding =utf-8
import caffe
import sys
import numpy as np
model_dir ="../model/caffe"

# 使输出的参数完全显示
# 若没有这一句，因为参数太多，中间会以省略号“……”的形式代替
np.set_printoptions(threshold=sys.maxsize)
maps ={"Pnet":1,"Rnet":2,"Onet":3}

def caffe2light(netname):
    print(netname)
    deploy_file=model_dir+"/det"+str(maps[netname])+".prototxt"
    caffe_model=model_dir+"/det"+str(maps[netname])+".caffemodel"
    net=caffe.Net(deploy_file,caffe_model,caffe.TEST)
    params_txt = "../model/light/"+netname+'.txt'
    pf = open(params_txt, 'w')
    for param_name in net.params.keys():
        print(param_name)
        try:
            weight = net.params[param_name][0].data
            shape = weight.shape    
            if len(weight.shape) == 4:
                width  = shape[3]
                height = shape[2]
                depth  = shape[1]
                amount = shape[0]
                for amountCount in range (0, amount):
                    if depth == 3:
                        for depthCount in range(depth-1,-1,-1):
                            for widthCount in range (0,width):
                                for heightCount in range (0,height):
                                    pf.write('[%.8f]\n' % net.params[param_name][0].data[amountCount][depthCount][heightCount][widthCount])
                    else:
                        for depthCount in range(0,depth):
                            for widthCount in range (0, width):
                                for heightCount in range (0, height):
                                    pf.write('[%.8f]\n' % net.params[param_name][0].data[amountCount][depthCount][heightCount][widthCount])
            else:
                weight.shape = (-1, 1)
                for w in weight:
                    pf.write('[%.8f]\n' % w)
        except:
            continue
        try:
            bias = net.params[param_name][1].data
            bias.shape = (-1, 1)
            for b in bias:
                pf.write('[%.8f]\n' % b)
        except:
            continue
        pf.close

def main():
    for m in maps:
        print(m)
        caffe2light(m)

if __name__=="__main__":
    main()