#!/usr/bin/env python3
"""
Initialize Yolo
"""

import tensorflow.keras as K
import numpy as np
import cv2
import os
import glob


class Yolo:
    """Yolo v3 algorithm for object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Class constructor"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as file:
            self.class_names = [line.strip() for line in file]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process the model outputs to extract bounding boxes,
        confidences, and class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size

        for i, output in enumerate(outputs):
            grid_height, grid_width = output.shape[:2]

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x, c_y = np.meshgrid(np.arange(grid_width),
                                   np.arange(grid_height))

            c_x = np.expand_dims(c_x, axis=-1)
            c_y = np.expand_dims(c_y, axis=-1)

            bx = (self.sigmoid(t_x) + c_x) / grid_width
            by = (self.sigmoid(t_y) + c_y) / grid_height
            bw = (np.exp(t_w) * self.anchors[i,
                  :, 0]) / self.model.input.shape[1]
            bh = (np.exp(t_h) * self.anchors[i,
                  :, 1]) / self.model.input.shape[2]

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters the bounding boxes.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            box_classes_i = np.argmax(scores, axis=-1)
            box_scores_i = np.max(scores, axis=-1)
            mask = box_scores_i >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(box_classes_i[mask])
            box_scores.append(box_scores_i[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return filtered_boxes, box_classes, box_scores

    def _iou(self, box1, box2):
        """
        Intersection Over Union of two bounding boxes
        """
        x1 = np.maximum(box1[0], box2[:, 0])
        y1 = np.maximum(box1[1], box2[:, 1])
        x2 = np.minimum(box1[2], box2[:, 2])
        y2 = np.minimum(box1[3], box2[:, 3])

        inter_area = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applying non-max suppression
        """
        indices = np.lexsort((-box_scores, box_classes))
        filtered_boxes = filtered_boxes[indices]
        box_scores = box_scores[indices]
        box_classes = box_classes[indices]

        unique_classes = np.unique(box_classes)
        nms_boxes = []
        nms_classes = []
        nms_scores = []

        for cls in unique_classes:
            cls_indices = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            while len(cls_boxes) > 0:
                max_score_index = np.argmax(cls_scores)
                nms_boxes.append(cls_boxes[max_score_index])
                nms_classes.append(cls)
                nms_scores.append(cls_scores[max_score_index])

                if len(cls_boxes) == 1:
                    break

                cls_boxes = np.delete(cls_boxes, max_score_index, axis=0)
                cls_scores = np.delete(cls_scores, max_score_index)

                ious = self._iou(nms_boxes[-1], cls_boxes)
                iou_indices = np.where(ious <= self.nms_t)[0]

                cls_boxes = cls_boxes[iou_indices]
                cls_scores = cls_scores[iou_indices]

        return (np.array(nms_boxes),
                np.array(nms_classes),
                np.array(nms_scores))

    @staticmethod
    def load_images(folder_path):
        """
        Load images
        """
        image_paths = glob.glob(os.path.join(folder_path, "*"))
        images = [cv2.imread(image_path) for image_path in image_paths]
        return images, image_paths

    def preprocess_images(self, images):
        """
        Resizes and rescales the images before processeing
        """
        pimages = []
        image_shapes = []

        for image in images:
            h, w, c = image.shape
            image_shapes.append([h, w])

            input_h = self.model.input.shape[1]
            input_w = self.model.input.shape[2]
            resized_img = cv2.resize(image,
                                     dsize=(
                                         input_h,
                                         input_w),
                                     interpolation=cv2.INTER_CUBIC)

            resized_img = resized_img / 255.0

            pimages.append(resized_img)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all boundary boxes, class names, and box scores.
        """
        for box, cls, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(img=image,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=(255, 0, 0),
                          thickness=2)

            class_name = self.class_names[cls]
            score_text = f"{score:.2f}"
            text = f"{class_name} {score_text}"

            cv2.putText(img=image,
                        text=text,
                        org=(x1, y1 - 5),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(0, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        cv2.imshow(file_name, image)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists("detections"):
                os.makedirs("detections")
            save_path = os.path.join("detections", file_name)
            cv2.imwrite(save_path, image)

        cv2.destroyWindow(file_name)
