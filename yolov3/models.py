from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from typing import Dict,List,Tuple

from utils.parse_config import parse_model_config
from utils.utils import build_targets,  non_max_suppression

def create_modules(module_defs:List[str]):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]#首个channel
    module_list = nn.ModuleList()
    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2
            modules.add_module(
                f"conv_{module_i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))
        
        elif module_def["type"] == "shortcut":
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())
    
        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())
                
        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)    

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [int(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def["classes"])

            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes)
            modules.add_module(f"yolo_{module_i}", yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

'''
yolo_0@layer_82:
input_sz:[1, 255, 13, 13], output_sz:[1, 507, 85]
result_sz:[1, 507, 85]

yolo_1@layer_94:
input_sz:[1, 255, 26, 26], output_sz:[1, 2028, 85]
result_sz:[1, 2535, 85]

yolo_2@layer_106:
input_sz:[1, 255, 52, 52], output_sz:[1, 8112, 85]
result_sz:[1, 10647, 85]
'''
class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors:Tuple[int,int], num_classes:int):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.FloatTensor(anchors).reshape(3,2) #以(w,h)元组为元素的列表
        self.num_anchors:int = len(anchors)
        self.num_classes:int = num_classes

    #x: (bn,18,grid,grid)
    def forward_base(self, x, img_size):

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        IntTensor = torch.cuda.IntTensor if x.is_cuda else torch.IntTensor

        num_anchors=IntTensor(self.num_anchors)
        num_classes=IntTensor(self.num_classes)
        num_samples = x.size(0)
        grid_size = x.size(2)
        stride = img_size / grid_size

        #(n,3,grid,grid,6)
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        
        x = torch.sigmoid(prediction[..., 0])  # Center x :(n,3,grid,grid)
        y = torch.sigmoid(prediction[..., 1])  # Center y :(n,3,grid,grid)
        w = prediction[..., 2]  # Width: (n,3,grid,grid)
        h = prediction[..., 3]  # Height:(n,3,grid,grid)
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf:(n,3,grid,grid)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred:(n,3,grid,grid,1)


        g=grid_size
        grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)#(1,1,g,g)
        grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)

        anchor_w = (self.anchors[:,0]/stride).view((1, self.num_anchors, 1, 1)).type(FloatTensor)
        anchor_h = (self.anchors[:,1]/stride).view((1, self.num_anchors, 1, 1)).type(FloatTensor)
        

        pred_boxes=torch.ones_like(prediction[..., :4])
        pred_boxes[..., 0] = x + grid_x#(bn,3,g,g)+(1,1,g,g) 广播
        pred_boxes[..., 1] = y + grid_y
        pred_boxes[..., 2] = w.exp_().mul_(anchor_w) #anchor_w@(1,3,1,1)，已经过缩放
        pred_boxes[..., 3] = h.exp_().mul_(anchor_h)

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        return output 

    #这个函数相较于上面的函数，把所有的张量写入操作进行了替换. 否则无法通过jit.trace.
    def forward(self, x, img_size:int):
        num_samples :int = x.size(0)
        grid_size :int = x.size(2)
        stride :int= img_size / x.size(2)

        #(n,3,grid,grid,6)
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        
        x = torch.sigmoid(prediction[..., 0])  # Center x :(n,3,grid,grid)
        y = torch.sigmoid(prediction[..., 1])  # Center y :(n,3,grid,grid)
        w = prediction[..., 2]  # Width: (n,3,grid,grid)
        h = prediction[..., 3]  # Height:(n,3,grid,grid)
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf:(n,3,grid,grid)
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred:(n,3,grid,grid,num_cls)
        pred_cls = pred_conf.unsqueeze(4)*pred_cls #按照kcacp后处理的的需要，这里对类别概率再乘以置信


        g=grid_size
        # torch.arange不能被trace. 使用下面的troch.cumsum替换.
        # grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).expand(num_samples,3,g,g).type(FloatTensor)#(1,1,g,g)
        # grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).expand(num_samples,3,g,g).type(FloatTensor)

        grid_x = torch.cumsum(torch.ones(g),0,dtype=torch.long)-1
        grid_y = torch.cumsum(torch.ones(g),0,dtype=torch.long)-1
        grid_x = grid_x.repeat(g, 1).view([1, 1, g, g]).expand(num_samples,3,g,g).type(x.type())#(1,1,g,g)
        grid_y = grid_y.repeat(g, 1).t().view([1, 1, g, g]).expand(num_samples,3,g,g).type(x.type())

        anchor_w = (self.anchors[:,0]/stride).view((1, self.num_anchors, 1, 1)).type(x.type())
        anchor_h = (self.anchors[:,1]/stride).view((1, self.num_anchors, 1, 1)).type(x.type())
        
        a= x + grid_x
        b= y + grid_y
        c=w.exp_().mul_(anchor_w)
        d=h.exp_().mul_(anchor_h)
        pred_boxes=torch.cat([a.unsqueeze_(-1),b.unsqueeze_(-1),c.unsqueeze_(-1),d.unsqueeze_(-1)],-1)

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        return output 


from torch.jit import Final

class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, config_path:str, img_size:int=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path) #读文件解析网络结构
        self.module_list = create_modules(self.module_defs)#创建网络模型
        self.img_size:int = img_size 
        self.seen:int = 0
        self.header_info = torch.IntTensor(np.array([0, 0, 0, self.seen, 0], dtype=np.int32))
        self.data_type = torch.nn.Parameter(torch.randn(1))

    def forward(self, x, original_shape):
        x=x.type(self.data_type.type())
        original_shape=original_shape.type(x.type())


        input_size :int= x.shape[2]
        layer_outputs = []
        yolo_outputs  = []
        for _, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def["type"] in ["convolutional", "upsample"]:
                x = module(x)
            elif module_def["type"] == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]            
            elif module_def["type"] == "route":
                x = torch.cat([layer_outputs[int(layer_i)] for layer_i in module_def["layers"].split(",")], 1)
            elif module_def["type"] == "yolo":
                x = module[0](x, input_size) 
                yolo_outputs.append(x)# 记录yolo的每层输出
            
            layer_outputs.append(x) # 记录每层的输出，供shortcut和route使用

        yolo_outputs = torch.cat(yolo_outputs, 1) # 把3层yolo输出在1维进行连接，最后形成@(1,10647,5+num_cls)输出
        yolo_outputs = self.rescale_boxes(yolo_outputs,original_shape) #进行缩放
        return yolo_outputs

    #     detections@sz(n,10647,5+num_cls)
    # original_shape@sz(n,2) 2:w,h
    # 这个函数不能通过 jit.trace，原因可能是对张量进行了写入操作. 属于不支持的操作.
    # 使用下一个函数进行替换
    def rescale_boxes_base(self,detections,original_shape):
        detections[...,:4]=detections[...,:4]/self.img_size
        #print('detections[...,:4] shape is:\n',detections[...,:4].shape)
        scaled_x = detections[...,0].t() * original_shape[:,0]
        scaled_y = detections[...,1].t() * original_shape[:,1]
        scaled_w = detections[...,2].t() * original_shape[:,0]
        scaled_h = detections[...,3].t() * original_shape[:,1]
        scaled = torch.cat([scaled_x.t().unsqueeze(2),scaled_y.t().unsqueeze(2),scaled_w.t().unsqueeze(2),scaled_h.t().unsqueeze(2)],dim=2)#batch,pred_n,4

        #print('scaled shape is:\n',scaled.shape)

        scaled[...,0]=scaled[...,0] - scaled[...,2]/2
        scaled[...,1]=scaled[...,1] - scaled[...,3]/2
        scaled[...,2]=scaled[...,0] + scaled[...,2]
        scaled[...,3]=scaled[...,1] + scaled[...,3]
        
        # print('detections is:\n',detections)
        # print('scaled is:\n',scaled)
        
        detections[...,:4]=scaled
    
        print('detections shape is:\n',detections.shape)

        return detections
    
    #这个函数相较于上面把对张量索引再写入的操作进行全部替换
    def rescale_boxes(self,detections,original_shape):
        first_half_scaled=detections[...,:4]/self.img_size
        scaled_x = first_half_scaled[...,0].t() * original_shape[:,0]
        scaled_y = first_half_scaled[...,1].t() * original_shape[:,1]
        scaled_w = first_half_scaled[...,2].t() * original_shape[:,0]
        scaled_h = first_half_scaled[...,3].t() * original_shape[:,1]
        scaled = torch.cat([scaled_x.t().unsqueeze(2),scaled_y.t().unsqueeze(2),scaled_w.t().unsqueeze(2),scaled_h.t().unsqueeze(2)],dim=2)#batch,pred_n,4


        x_min=scaled[...,0] - scaled[...,2]/2
        y_min=scaled[...,1] - scaled[...,3]/2
        x_max=x_min + scaled[...,2]
        y_max=y_min + scaled[...,3]
        
        first=torch.cat([x_min.unsqueeze(2),y_min.unsqueeze(2),x_max.unsqueeze(2),y_max.unsqueeze(2)],dim=2) 
        output=torch.cat([first,detections[...,4:]],-1)

        return output

    def load_darknet_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
            self.header_info = header  # Needed to write header when saving weights
            self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w


