import argparse
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages,\
    letterbox


#USEFUL FOR KNOWING THE CONFIG
# def make_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     return parser


def detect(image):
    # HARDCODED SETTINGS
    weights = 'data/weights/yolopv2.pt'
    imgsz = 640
    device = '0'
    classes = None
    agnostic = False
    conf_thres = 0.3
    iou_thres=0.45

    # Load model
    stride = 32
    model  = torch.jit.load(weights)
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16  
    model.eval()

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    random_bgr = np.transpose(random_image, (1, 2, 0))

    img0 = cv2.resize(random_bgr, (1280,720), interpolation=cv2.INTER_LINEAR)
    img = letterbox(img0)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    t1 = time_synchronized()
    [pred,anchor_grid],seg,ll= model(img)
    t2 = time_synchronized()

    # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
    # but this problem will not appear in offical version 
    tw1 = time_synchronized()
    pred = split_for_trace_model(pred,anchor_grid)
    tw2 = time_synchronized()

    # Apply NMS
    t3 = time_synchronized()
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic)
    t4 = time_synchronized()

    da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)

    show_seg_result(img0, (da_seg_mask,ll_seg_mask), is_demo=True)
    return img0


if __name__ == '__main__':
    with torch.no_grad():
            #HERE SHOULD GO THE IMAGE FROM CARLA
            random_image = np.random.randint(0, 256, size=(3, 600, 800), dtype=np.uint8)
            
            start_time = time.time()
            img = detect(random_image)
            end_time = time.time()
            
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds")
            plt.imshow(img)
            plt.axis('off')  # Hide axes
            plt.show()