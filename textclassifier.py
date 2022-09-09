import sys

sys.path.insert(0, './yolov7')
import torch
import numpy as np
import cv2
import os
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.datasets import letterbox


class TextClassifier:

    def __init__(self, weights, device='cpu'):
        self.dataset = None
        self.weights = weights
        self.device = select_device(device)
        self.predictor = attempt_load(weights, map_location=device)

    def create_dataset(self, data_path):
        self.dataset = LoadImages(data_path)
        return self.dataset

    def preprocess_img(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def predict(self, img, colors, conf_thresh=0.25, iou_thresh=0.45, filter_classes=None):
        im0s = letterbox(img.copy(), (640, 640), stride=32)[0]

        # Convert
        im0s = im0s[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        im0s = np.ascontiguousarray(im0s)

        with torch.no_grad():
            im0s = self.preprocess_img(im0s)
            predictions = self.predictor(im0s)[0]
            predictions = non_max_suppression(predictions, conf_thres=conf_thresh, iou_thres=iou_thresh,
                                              classes=filter_classes)
            final_predictions = []
            for boxes in predictions:
                boxes[:, :4] = scale_coords(im0s.shape[2:], boxes[:, :4], img.shape).round()

                for box in boxes:
                    xyxy = box[:4]
                    conf, cls = box[4], box[5]
                    xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                    name = self.predictor.names[int(cls)]
                    line = [name, int(cls), int(xywh[0]), int(xywh[1]), int(xywh[2]),
                            int(xywh[3]), float(conf)]
                    label = f'{self.predictor.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=1)
                    final_predictions.append(line)
        return img, final_predictions

    def batch_predict(self, dataset, out_path, colors, conf_thresh=0.25, iou_thresh=0.45, filter_classes=None):
        with torch.no_grad():
            cls_names = self.predictor.names
            out_label_path = os.path.join(out_path, 'labels')
            out_image_path = os.path.join(out_path, 'images')
            if not os.path.exists(out_image_path):
                os.makedirs(out_image_path)
            if not os.path.exists(out_label_path):
                os.makedirs(out_label_path)
            print("Please wait: Processing..")
            for path, img, im0, _ in dataset:
                im0s = im0.copy()
                img_name = path.split('/')[-1]
                out_img_path = os.path.join(out_image_path, img_name)
                out_txt_path = os.path.join(out_label_path, os.path.splitext(img_name)[0] + '.txt')
                img = self.preprocess_img(img)
                predictions = self.predictor(img)[0]
                predictions = non_max_suppression(predictions, conf_thres=conf_thresh, iou_thres=iou_thresh,
                                                  classes=filter_classes)

                for boxes in predictions:
                    boxes[:, :4] = scale_coords(img.shape[2:], boxes[:, :4], im0s.shape).round()

                    for box in boxes:
                        xyxy = box[:4]
                        conf, cls = box[4], box[5]
                        xywh = [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]
                        line = [str(cls.numpy()), str(xywh[0].numpy()), str(xywh[1].numpy()), str(xywh[2].numpy()),
                                str(xywh[3].numpy()), str(conf.numpy())]
                        with open(out_txt_path, 'a') as f:
                            f.write(' '.join(line) + '\n')
                        label = f'{cls_names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=1)

                cv2.imwrite(out_img_path, im0s)
            print("Predictions were saved in:", out_path)
