import sys
sys.path.append('./')
sys.path.append('./yolov3')
sys.path.append('./facenet')

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from yolov3.models import *
from yolov3.utils.utils import *
from yolov3.utils.datasets import *
from trace_model_utils.gen_traced_model import gen_cpu_traced_model,gen_gpu_traced_model

from resnet100.mx2pytorch_resnet100 import KitModel as ResnetModel
from mobilefacenet.mx2pytorch_mobileface import KitModel as MobileFace
from clear.tf2pytorch_clear import KitModel as ClearModel
from mtcnn.mtcnn_model import PNet,RNet,ONet
#from facenet.pytorch_facenet import InceptionResnetV1

def trace_yolov3():
    model_def='yolov3/config/kcyolov3.cfg'
    img_size=416
    weights_path='yolov3/weights/kc_yolov3_best_delta.weights'

    model = Darknet(model_def, img_size=img_size)
    model.load_darknet_weights(weights_path)

    batch_sz=4
    inputs=torch.randn(batch_sz,3,416,416)
    intpus_sz=torch.randn(batch_sz,2)
    out=model(inputs,intpus_sz)

    #cpu-checked-ok
    print('cpu:')
    traced_cpu=gen_cpu_traced_model(model,[(10,3,416,416),(10,2)],'yolov3')
    batch_sz+=3
    inputs=torch.randn(batch_sz,3,448,448)#
    intpus_sz=torch.randn(batch_sz,2)
    out=traced_cpu(inputs,intpus_sz)
    print(out.shape)

    #gpu-checked-ok
    print('gpu:')
    traced_gpu=gen_gpu_traced_model(model,[(10,3,416,416),(10,2)],'yolov3')
    batch_sz+=3
    inputs=torch.randn(batch_sz,3,448,448)
    intpus_sz=torch.randn(batch_sz,2)
    out=traced_gpu(inputs,intpus_sz)
    print(out.shape)

def trace_yolov3_coco():
    model_def='yolov3/config/yolov3.cfg'
    img_size=416
    weights_path='yolov3/weights/yolov3.weights'

    model = Darknet(model_def, img_size=img_size)
    model.load_darknet_weights(weights_path)

    batch_sz=4
    inputs=torch.randn(batch_sz,3,416,416)
    intpus_sz=torch.randn(batch_sz,2)
    out=model(inputs,intpus_sz)

    #cpu-checked-ok
    print('cpu:')
    traced_cpu=gen_cpu_traced_model(model,[(10,3,416,416),(10,2)],'yolov3-coco')
    batch_sz+=3
    inputs=torch.randn(batch_sz,3,448,448)#
    intpus_sz=torch.randn(batch_sz,2)
    out=traced_cpu(inputs,intpus_sz)
    print(out.shape)

    #gpu-checked-ok
    print('gpu:')
    traced_gpu=gen_gpu_traced_model(model,[(10,3,416,416),(10,2)],'yolov3-coco')
    batch_sz+=3
    inputs=torch.randn(batch_sz,3,448,448)
    intpus_sz=torch.randn(batch_sz,2)
    out=traced_gpu(inputs,intpus_sz)
    print(out.shape)    


def trace_resnet100():
    model=ResnetModel('resnet100/npy_weights/weights.npy')
    inputs=torch.randn(3,3,112,112)
    out=model(inputs)
    
    #cpu-checked-ok
    print('cpu :')
    traced_cpu=gen_cpu_traced_model(model,[(2,3,112,112)],'resnet100',True)
    inputs=torch.randn(3,3,112,112)
    out=traced_cpu(inputs)
    print(out.shape)
    
    #gpu-checked-ok
    print('gpu :')
    traced_gpu=gen_gpu_traced_model(model,[(2,3,112,112)],'resnet100',True)
    out=traced_gpu(inputs)
    print(out.shape)

def trace_mobileface():
    model=MobileFace('mobilefacenet/npy_weights/weights.npy')
    inputs=torch.randn(2,3,80,80)
    out=model(inputs)
    
    #cpu-checekd-ok
    print('cpu:')
    traced_cpu=gen_cpu_traced_model(model,[(2,3,80,80)],'mobileface')
    inputs=torch.randn(3,3,80,80)
    out=traced_cpu(inputs)
    print(out.shape)

    #gpu-checked-ok
    print('gpu:')
    traced_gpu=gen_gpu_traced_model(model,[(2,3,80,80)],'mobileface')
    inputs=torch.randn(3,3,80,80)
    out=traced_gpu(inputs)
    print(out.shape)

def trace_clearcls():
    model=ClearModel('clear/npy_weights/weights.npy')
    inputs=torch.randn(2,3,48,48)
    out=model(inputs)
    
    #cpu-checked-ok
    print('cpu:')
    traced_cpu=gen_cpu_traced_model(model,[(2,3,48,48)],'clear')
    inputs=torch.randn(5,3,48,48)
    out=traced_cpu(inputs)
    print(out.shape)

    #gpu-checked-ok
    print('gpu:')
    traced_gpu=gen_gpu_traced_model(model,[(2,3,48,48)],'clear')
    out=traced_gpu(inputs)
    print(out.shape)


def trace_mtcnn():
    pnet_weights_path='mtcnn/mtcnn_det1.npy'
    rnet_weights_path='mtcnn/mtcnn_det2.npy'
    onet_weights_path='mtcnn/mtcnn_det3.npy'

    batch_sz=2
    scale=torch.FloatTensor([0.5])

##pnet
    print('pnet:')
    p=PNet(pnet_weights_path)
    #cpu-checked-ok
    print('cpu:')
    p_cpu=gen_cpu_traced_model(p,[(batch_sz,3,12,12),1],'pnet')
    out=p_cpu(torch.randn(7,3,12,12),scale)
    print(out.shape)
    
    #gpu-checked-ok
    print('gpu:')
    p_gpu=gen_gpu_traced_model(p,[(batch_sz,3,12,12),1],'pnet')
    out=p_gpu(torch.randn(7,3,12,12),scale)
    print(out.shape)

##rnet
    print('rnet:')
    r=RNet(rnet_weights_path)
    #cpu-checked-ok
    print('cpu:')
    r_cpu=gen_cpu_traced_model(r,[(batch_sz,3,24,24),(batch_sz,2)],'rnet')
    out=r_cpu(torch.randn(5,3,24,24),torch.randn(5,2))
    print(out.shape)

    #gpu-checked-ok
    print('gpu:')
    r_gpu=gen_gpu_traced_model(r,[(batch_sz,3,24,24),(batch_sz,2)],'rnet')
    out=r_gpu(torch.randn(6,3,24,24),torch.randn(6,2))
    print(out.shape)

    
##onet

    print('onet:')
    o=ONet(onet_weights_path)
    
    #cpu-checked-ok
    print('cpu:')
    o_cpu=gen_cpu_traced_model(o,[(batch_sz,3,48,48),(batch_sz,2)],'onet')
    out=o_cpu(torch.randn(7,3,48,48),torch.randn(7,2))
    print(out.shape)

    #gpu-checked-ok
    print('gpu:')
    o_gpu=gen_gpu_traced_model(o,[(batch_sz,3,48,48),(batch_sz,2)],'onet')
    out=o_gpu(torch.randn(8,3,48,48),torch.randn(8,2))
    print(out.shape)

def trace_facenet():
    facenet_pt='facenet/facenet.pt'
    facenet = torch.load(facenet_pt)
    inputs=torch.randn(2,3,160,160)
    out=facenet(inputs)

    #cpu-checked-ok
    print('cpu:')
    traced_cpu=gen_cpu_traced_model(facenet,[(2,3,160,160)],'facenet')
    inputs=torch.randn(5,3,160,160)
    out=traced_cpu(inputs)
    print(out.shape)

    #gpu-checked-ok
    print('gpu:')
    traced_gpu=gen_gpu_traced_model(facenet,[(2,3,160,160)],'facenet')
    out=traced_gpu(inputs)
    print(out.shape)



def get_model_inputs(name,batch_sz,dev):
    if name == 'yolov3':
        yolov3_inputs=torch.randn(batch_sz,3,416,416)
        yolov3_intpus_shape=torch.randn(batch_sz,2)
        return yolov3_inputs.to(dev),yolov3_intpus_shape.to(dev)

    if name == 'res100':
        res100_inputs=torch.randn(batch_sz,3,112,112)
        return res100_inputs.to(dev)

    if name == 'mobileface':
        mobileface_inputs=torch.randn(batch_sz,3,80,80)
        return mobileface_inputs.to(dev)

    if name == 'clear':
        clear_inputs=torch.randn(batch_sz,3,48,48)
        return clear_inputs.to(dev)

    if name == 'pnet':
        pnet_inputs=torch.randn(batch_sz,3,12,12)
        scale=torch.randn(1)
        return pnet_inputs.to(dev),scale.to(dev)

    if name == 'rnet':
        rnet_inputs=torch.randn(batch_sz,3,24,24)
        rnet_shape=torch.randn(batch_sz,4)
        return rnet_inputs.to(dev),rnet_shape.to(dev)

    if name == 'onet':
        onet_inputs=torch.randn(batch_sz,3,48,48)
        onet_shape=torch.randn(batch_sz,4)
        return onet_inputs.to(dev),onet_shape.to(dev)

    if name == 'facenet':
        fancenet_inputs=torch.randn(batch_sz,3,160,160)
        return fancenet_inputs.to(dev)


    raise RuntimeError('model name WRONG!')        



#All-models-In-One
class AIO(torch.nn.Module):
    def __init__(self):
        super().__init__()
        yolov3_model_def='yolov3/config/kcyolov3.cfg'
        yolov3_img_size=416
        yolov3_weights_path='yolov3/weights/kc_yolov3_best_delta.weights'
        res100_weights_path='resnet100/npy_weights/weights.npy'
        mobileface_weights_path='mobilefacenet/npy_weights/weights.npy'
        clear_weights_path='clear/npy_weights/weights.npy'
        pnet_weights_path='mtcnn/mtcnn_det1.npy'
        rnet_weights_path='mtcnn/mtcnn_det2.npy'
        onet_weights_path='mtcnn/mtcnn_det3.npy'
        facenet_pt='facenet/facenet.pt'
        yolov3_model = Darknet(yolov3_model_def, img_size=yolov3_img_size)
        yolov3_model.load_darknet_weights(yolov3_weights_path)
        facenet = torch.load(facenet_pt)
        
        self.yolov3=yolov3_model        
        self.res100=ResnetModel(res100_weights_path)
        self.mobileface=MobileFace(mobileface_weights_path)
        self.clear=ClearModel(clear_weights_path)
        self.pnet=PNet(pnet_weights_path)
        self.rnet=RNet(rnet_weights_path)
        self.onet=ONet(onet_weights_path)
        self.facenet=facenet


    def yolov3_forward(self,inputs,inputs_shape):
        out=self.yolov3(inputs,inputs_shape)
        return out

    def res100_forward(self,inputs):
        out=self.res100(inputs)
        return out

    def mobileface_forward(self,inputs):
        out=self.mobileface(inputs)
        return out

    def clear_forward(self,inputs):
        out=self.clear(inputs)
        return out

    def pnet_forward(self,inputs,scale):
        out=self.pnet(inputs,scale)
        return out

    def rnet_forward(self,inputs,inputs_shape):
        out=self.rnet(inputs,inputs_shape)
        return out

    def onet_forward(self,inputs,inputs_shape):
        out=self.onet(inputs,inputs_shape)
        return out

    def facenet_forward(self,inputs):
        out=self.facenet(inputs)
        return out

    def forward(self,x):
        return x


#把所有模型合并在一起
def trace_all(cuda):
    model = AIO()
    model.eval()
    if cuda:
        dev=torch.device('cuda')
    else:
        dev=torch.device('cpu')

    #dev=torch.device('cpu') if cpu else torch.device('cuda')
    model.to(dev)
    
    # test OK.
    # print(model.mobileface.fc1.weight.type())
    # print(model.onet.conv6_1.weight.type())
    # return

    with torch.no_grad():
        yolov3_inputs,yolov3_intpus_shape=get_model_inputs('yolov3',4,dev)
        res100_inputs=get_model_inputs('res100',4,dev)
        mobile_inputs=get_model_inputs('mobileface',4,dev)
        clear_inputs=get_model_inputs('clear',4,dev)
        pnet_inputs,scale=get_model_inputs('pnet',4,dev)
        rnet_inputs,rnet_inputs_shape=get_model_inputs('rnet',4,dev)
        onet_inputs,onet_inputs_shape=get_model_inputs('onet',4,dev)
        facenet_inputs=get_model_inputs('facenet',4,dev)

        traced_map={'yolov3_forward':(yolov3_inputs,yolov3_intpus_shape),
        'res100_forward':res100_inputs,
        'mobileface_forward':mobile_inputs,
        'clear_forward':clear_inputs,
        'pnet_forward':(pnet_inputs,scale),
        'rnet_forward':(rnet_inputs,rnet_inputs_shape),
        'onet_forward':(onet_inputs,onet_inputs_shape),
        'facenet_forward':facenet_inputs
        }

        #with torch.jit.optimized_execution(True):#这个优化context看不出什么效果
        traced_model=torch.jit.trace_module(model,traced_map,check_trace=False)
        traced_model.eval()
        
        yolov3_inputs,yolov3_intpus_shape=get_model_inputs('yolov3',5,dev)
        res100_inputs=get_model_inputs('res100',6,dev)
        mobile_inputs=get_model_inputs('mobileface',7,dev)
        clear_inputs=get_model_inputs('clear',8,dev)
        pnet_inputs,scale=get_model_inputs('pnet',9,dev)
        rnet_inputs,rnet_inputs_shape=get_model_inputs('rnet',10,dev)
        onet_inputs,onet_inputs_shape=get_model_inputs('onet',11,dev)
        facenet_inputs=get_model_inputs('facenet',12,dev)


        print('yolov3:')
        out=traced_model.yolov3_forward(yolov3_inputs,yolov3_intpus_shape)
        print(out.shape)

        print('\nres100:')
        out=traced_model.res100_forward(res100_inputs)
        print(out.shape)

        print('\nclear:')
        out=traced_model.clear_forward(clear_inputs)
        print(out.shape)
        print(out)

        print('\nmobileface:')
        out=traced_model.mobileface_forward(mobile_inputs)
        print(out.shape)

        print('\npnet:')
        out=traced_model.pnet_forward(pnet_inputs,scale)
        print(out.shape)

        print('\nrnet:')
        out=traced_model.rnet_forward(rnet_inputs,rnet_inputs_shape)
        print(out.shape)

        print('\nonet')
        out=traced_model.onet_forward(onet_inputs,onet_inputs_shape)
        print(out.shape)

        print('\nfacenet')
        out=traced_model.facenet_forward(facenet_inputs)
        print(out.shape)


        if not cuda:
            traced_model.save('aio_cpu.pt')
        else:
            traced_model.save('aio_gpu.pt')


if __name__=='__main__':
    if torch.__version__ != "1.3.1+cu100":
        print('please install correct pytorch versinon: 1.3.1+cu100')
        print('You may get the package here:\\\\192.168.16.42\\share\\haihong.qin\\wheels\\torch-1.3.1+cu100-cp36-cp36m-linux_x86_64.whl')
        sys.exit(0)
    
    #1.生成各自独立的模型
    #trace_yolov3()
    #trace_yolov3_coco()
    #trace_resnet100()
    #trace_mobileface()
    #trace_clearcls()
    #trace_mtcnn()
    #trace_facenet()

    #2.生成整合模型
    print('gpu jit trace model:')
    trace_all(cuda=True)
    print('*'*20)
    print('cpu jit trace model:')
    trace_all(cuda=False)
