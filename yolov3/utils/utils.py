from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import random

def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    with open(path,'r') as f:
        lines=f.readlines()
        lines=[x for x in lines if len(x)>0]
        lines=[x.lstrip().rstrip() for x in lines]
    return lines


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

#注意：传入的参数是在训练集上的总的统计量
#true_positives@ndarray(n,), 置信@ndarray(n,)，预测类别@ndarray(n,)
#target_cls是训练集对应的gt标签@list(tn,)
def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    # 按置信降序排列
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    # 对所有的gt label进行unique处理
    unique_classes = np.unique(target_cls)#(c,)

    # Create Precision-Recall curve and compute AP for each class
    # 为每个 gt 类别计算precision，recall，AP
    # 注意 precesion-recall 曲线是 true_postive和false_positive各自进行累加，再除以特定分母,形成的变化的区间列表，代表x轴和y轴
    # 而precision和recall则是各自列表的最后一项值，表示最终的计算结果.
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects. :ctn
        n_p = i.sum()  # Number of predicted objects. :cpn

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()#(cpn,)
            tpc = (tp[i]).cumsum()#(cpn,)

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)#(cpn,)
            r.append(recall_curve[-1])#只把最后一项计算的总的recall挂到r

            # Precision
            precision_curve = tpc / (tpc + fpc)#(cpn,)
            p.append(precision_curve[-1])#只把最后一项计算的总的精度挂到p

            # AP from recall-precision curve
            # 使用AUC AP计算
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    # p,r,ap -> (c,1)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

#计算AUC AP . 参考：http://note.youdao.com/noteshare?id=2c24d33f527df44b2c057cffd7954ba1
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end 0.添加哨兵
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope 1.逆向计算precision包络
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points 2.寻找recall的改变点
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec 3.计算AUC：对recall的所有改变区间与包络precision的乘积进行加和
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

#output@list(8,) 
#targets@tensor(t,6)，已经包括这个批内所有图像的gt box,通过第一列的标记，来区分批内不同的图像索引
#iou_threshold@float(0.5,)
def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue
        #outputs已经过nms处理
        output = outputs[sample_i]#当前图像的预测输出：(n,7)
        pred_boxes = output[:, :4]#(n,4)
        pred_scores = output[:, 4]#(n,)
        pred_labels = output[:, -1]#(n,)
        #true_positve的数量等同于所有的预测框数量
        true_positives = np.zeros(pred_boxes.shape[0])#(n,)
        #通过第0列的标记，找到对应当前图像包含的所有gt boxes
        annotations = targets[targets[:, 0] == sample_i][:, 1:]#(tl,5)
        target_labels = annotations[:, 0] if len(annotations) else []#(tl,)
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]#(tl,4)
            #遍历当前图像的所有预测框和预测标签
            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue
                #对当前预测框，找到与之IOU最大的gt box，作为true positvie
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)#(1,tl)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])#每个图像统计tp,conf,label：@tensor(n,)
    return batch_metrics

#shape iou
def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

#prediction(bn,10647,85)->output[]:bn,item(_m,7)@tensor
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4,ltrb=True):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    if not ltrb:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]#批内索引
    ##原来的nms使用3个循环，下面以两个循环解决.
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres] #(_n,85)
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0] #乘号右边先执行
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)#(_n,7)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            #这个large_overlap是超过nms_threshold、需要过滤掉的
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres #(_n,)
            label_match = detections[0, -1] == detections[:, -1] #(_n,)
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match #(_n,)
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            # 这里对所有同类的、IOU大于nms阈值的无效预测，取加权平均，作为最终的预测, 因此会原则上会更好一些
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

'''
这一版改的晦涩难懂，虽然可能提升了性能，但远不如第一版那么清晰易懂.

pred_boxes:(n,3,grid,grid,4)
pred_cls:(n,3,grid,grid,80)
target:(gt_n,6) #这一批内所有的gt. 坐标为相对于原图的中心-宽高比例形式. 
anchors: (3,2),3个原始anchor,经 416/stride 缩放
ignore_thres:0.5
'''
def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    '''
    根据  预测@(b,3,g,g,[c]) 构造
    对应  标签@(b,3,g,g,[c])
    由于并不是所有预测cell都有实际对应的gt，所以要根据gt，得到相应的mask，通过mask来确定某个cell不是不有对应gt.
    '''
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    #pred_boxes只用于提取对应于训练数据的信息
    nB = pred_boxes.size(0) #batch
    nA = pred_boxes.size(1) #anchor
    nC = pred_cls.size(-1)  #class num
    nG = pred_boxes.size(2) #grid

    # Output tensors
    # t字头包括mask的尺度，都是对应于prediction的尺度@(b,3,g,g,[c])
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)#同时obj_mask(转为float)也相当于tconf
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    #两个测度，计算预测类别的准确率以及iou，不参与计算损失.
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)

    # Convert to position relative to box
    # 对方形填充过的gt进行缩放到grid_sz尺度
    target_boxes = target[:, 2:6] * nG #(gt_n,4) 中心宽高 形式
    gxy = target_boxes[:, :2] #(gt_n,2)
    gwh = target_boxes[:, 2:] #(gt_n,2)
    
    # Get anchors with best iou
    # 这是shape IOU：找到与gt最为匹配的anchor：相当于计算(0,0,anchor_w,anchor_y)与(0，0，gt_w,gt_w)之间的IOU
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])#(3,gt_n) 每行表示该anchor与所有gt的iou
    best_ious, best_n = ious.max(0)#(gtn,) (gtn,) 找到每个gt对应最大iou的anchor
    
    # Separate target values
    b, target_labels = target[:, :2].long().t()#batch:(gt_n,) label:(gt_n,)
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    
    #可以仿照faster-rcnn构造selected_indices
    #selected_indices=torch.stack([b, best_n, gj, gi])#(4,gt_n)

    # Set masks
    obj_mask[b, best_n, gj, gi] = 1 #(gt_n,)
    noobj_mask[b, best_n, gj, gi] = 0#(gt_n,)

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates 的逆变换和Faster-RCNN不同
    tx[b, best_n, gj, gi] = gx - gx.floor() #floor的结果和上面的long的结果相同，只是这里是float类型
    ty[b, best_n, gj, gi] = gy - gy.floor()

    # Width and height的逆变换和Faster-RCNN相同
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    
    # 用obj_mask表示tconf
    tconf = obj_mask.float()
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1 #(gt_n,)

    # Compute label correctness and iou at best anchor
    # 注意 pred_cls[b, best_n, gj, gi] 这种写法是多维数组索引. b, best_n, gj, gi 尺度都是gt_n.
    # target_labels 尺度也是gt_n
    # 有gt对应的预测cell(gt_n个)的预测类别，预测正确了多少个. 测度之一，不用来计算损失.
    # pred_cls[b, best_n, gj, gi] 多维索引，结果是@(gt_n,85)，然后再.argmax(-1)，选出每行概率最大的预测标签，结果是(gt_n,)个预测类的标签. 
    # 接着再与 target_labels 比较. 得到gt_n个逻辑索引赋值给 ：class_mask[b, best_n, gj, gi]@(gt_n,)
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    
    #有gt对应的cell(gt_n个)的预测坐标，与gt的IOU是多少. 测度之一，不用来计算损失.
    # pred_boxes[b, best_n, gj, gi]@(gt_n,4) target_boxes@(gt_n,4)
    # iou_scores[b, best_n, gj, gi]@(gt_n,)
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)
    

    # 所有返回t字头标签和mask，以及两个测度. 尺度都是@(b,3,g,g,[c]). c仅针对于tcls.
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

def visualize_detections(imgs_path,imgs_detection,yoloInputSize,classes,rescaled=True):
    os.makedirs("output", exist_ok=True)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs_path, imgs_detection)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            if not rescaled:
                detections = rescale_boxes(detections, yoloInputSize, img.shape[:2])#在这里对预测框做变换到原图尺寸.
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                
                print('cls_pred is:',cls_pred)
                print('classes is:',classes)
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        #filename = path.split("/")[-1].split(".")[0]
        base=os.path.basename(path).split('.')[0]
        plt.savefig("{}_output.png".format(base), bbox_inches="tight", pad_inches=0.0)
        plt.close()

