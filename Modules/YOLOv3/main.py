import os
import cv2

import sys
import time
from PIL import Image, ImageDraw
from Modules.YOLOv3.utils import *
from Modules.YOLOv3.darknet import Darknet

class YOLOv3:
    model = None
    result = None
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        # TODO
        #   - initialize and load model here
        cfg_path = os.path.join(self.path, "yolov3-nsfw.cfg")
        model_path = os.path.join(self.path, "yolov3-nsfw.weights")
        data_path = os.path.join(self.path, "nsfw.data")
        self.model = Darknet(cfg_path)

        self.model.print_network()
        self.model.load_weights(model_path)
        print('Loading weights from %s... Done!' % (model_path))

        self.namesfile = data_path

        self.use_cuda = 1
        if self.use_cuda:
            self.model.cuda()




    def inference_by_path(self, image_path):
        # TODO
        #   - Inference using image path
        image = cv2.imread(image_path)
        results = self.model.detect(image, rgb=False)

        img = Image.open(image_path).convert('RGB')
        sized = img.resize((self.model.width, self.model.height))

        for i in range(2):
            start = time.time()
            boxes = do_detect(self.model, sized, 0.5, 0.4, self.use_cuda)
            finish = time.time()
            if i == 1:
                print('%s: Predicted in %f seconds.' % (image_path, boxes, (finish - start)))

        result = []
        # for i, obj in enumerate(results):
        #     print(obj.name)
        #     xmin, ymin, xmax, ymax = obj.to_xyxy()
        #     x = xmin
        #     y = ymin
        #     w = xmax - xmin
        #     h = ymax - ymin
        #     result.append([(x,y,w,h), {obj.name: obj.prob}])

        self.result = result

        return self.result