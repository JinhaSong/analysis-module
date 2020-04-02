from Modules.YOLOv3.models import *
from Modules.YOLOv3.utils.utils import *
from Modules.YOLOv3.utils.datasets import *

import os
import time

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import cv2


class YOLOv3:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.config_path = os.path.join(self.path, "config", "yolov3-nsfw.cfg")
        self.weights_path = os.path.join(self.path, "weights", "yolov3-nsfw_50000.weights")
        self.class_path = os.path.join(self.path, "data", "nsfw.names")
        self.img_size = 604
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.batch_size = 1
        self.n_cpu = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Darknet(self.config_path, img_size=self.img_size).to(device)

        if self.weights_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.weights_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.weights_path))

        self.model.eval()  # Set in evaluation mode

        self.classes = load_classes(self.class_path)  # Extracts class labels from file
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def inference_by_path(self, file_path):
        fname, ext = os.path.splitext(file_path)
        zip_file = None
        if ext == ".zip" :
            loader = ZipFileLoader(file_path, img_size=self.img_size)
            zip_file = zipfile.ZipFile(file_path)
        elif ext == ".tar" :
            loader = TarLoader(file_path, img_size=self.img_size)
        else :
            loader = ImageLoader(file_path, img_size=self.img_size)

        dataloader = DataLoader(
            # TarLoader("D:\\Projects\\GitRepo\\VideoPreviewer\\yolov3\\test.tar", img_size=img_size),
            loader,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_cpu,
        )

        imgs = []
        img_detections = []

        prev_time = time.time()
        for batch_i, (input_name, input_imgs) in enumerate(dataloader):
            input_imgs = Variable(input_imgs.type(self.Tensor))

            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

            imgs.append(input_name)
            img_detections.extend(detections)



        results = []
        for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
            img_name = path[0]
            if ext == ".zip":
                img = zip_file.read(img_name)
                img = np.array(Image.open(io.BytesIO(img)))
            else :
                img = cv2.imread(img_name)

            shape = img.shape[:2]

            img_result = {
                "image_name": img_name,
                "result": []
            }
            if detections is not None:
                # Rescale boxes to original image
                detections = rescale_boxes(detections, self.img_size, shape)
                for x, y, x2, y2, conf, cls_conf, cls_pred in detections:
                    w = x2 - x
                    h = y2 - y
                    img_result['result'].append({
                        "position": {
                            "x": float(x),
                            "y": float(y),
                            "w": float(w),
                            "h": float(h),
                        },
                        "label": [{
                            "description": str(self.classes[int(cls_pred)]),
                            "score": float(cls_conf.item())
                        }]
                    })
            results.append(img_result)

        if ext == ".zip" :
            zip_file.close()
        self.result = results

        return self.result

