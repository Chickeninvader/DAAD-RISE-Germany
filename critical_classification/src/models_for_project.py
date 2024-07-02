import os.path
from collections import deque, defaultdict

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision
from keras.applications import MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, LeakyReLU, Dropout, Reshape, Lambda
from keras.models import Model
from transformers import VideoMAEForVideoClassification
from ultralytics import YOLO


class YOLOv1(nn.Module):
    """
    This class contains the YOLOv1 model. It consists of 24 convolutional and
    2 fully-connected layers which divide the input image into a
    (split_size x split_size) grid and predict num_boxes bounding boxes per grid
    cell.
    """

    def __init__(self, split_size, num_boxes, num_classes):
        """
        Initializes the neural-net with the parameter values to produce the
        desired predictions.

        Parameters:
            split_size (int): Size of the grid which is applied to the image.
            num_boxes (int): Amount of bounding boxes which are predicted per
            grid cell.
            num_classes (int): Amount of different classes which are being
            predicted by the model.
        """

        super(YOLOv1, self).__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.darkNet = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3, stride=2, bias=False),  # 3,448,448 -> 64,224,224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 64,112,112

            nn.Conv2d(64, 192, 3, padding=1, bias=False),  # -> 192,112,112
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 192,56,56

            nn.Conv2d(192, 128, 1, bias=False),  # -> 192,56,56
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 3, padding=1, bias=False),  # -> 256,56,56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, 1, bias=False),  # -> 256,56,56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),  # -> 512,56,56
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 512,28,28

            nn.Conv2d(512, 256, 1, bias=False),  # -> 256,28,28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),  # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, bias=False),  # -> 256,28,28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),  # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, bias=False),  # -> 256,28,28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),  # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, 1, bias=False),  # -> 256,28,28
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, 3, padding=1, bias=False),  # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, 1, bias=False),  # -> 512,28,28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),  # -> 1024,28,28
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # -> 1024,14,14

            nn.Conv2d(1024, 512, 1, bias=False),  # -> 512,14,14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),  # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, 1, bias=False),  # -> 512,14,14
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, 3, padding=1, bias=False),  # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),  # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),  # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),  # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1, bias=False),  # -> 1024,14,14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024 * split_size * split_size, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, split_size * split_size * (num_classes + num_boxes * 5)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forwards the input tensor through the model to produce the predictions.

        Parameters:
            x (tensor): A tensor of shape (batch_size, 3, 448, 448) which represents
            a batch of input images.

        Returns:
            x (tensor): A tensor of shape
            (batch_size, split_size, split_size, num_boxes*5 + num_classes)
            which contains the predicted bounding boxes.
        """

        x = self.darkNet(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = x.view(x.shape[0], self.split_size, self.split_size,
                   self.num_boxes * 5 + self.num_classes)
        return x


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class ResNet3D(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

        self.resnet_3d = torchvision.models.video.r3d_18()
        self.output_layer = nn.Sequential(
            nn.Linear(400, 64),  # First layer
            nn.LeakyReLU(0.1),
            nn.Linear(64, 8),  # Second layer
            nn.LeakyReLU(0.1),  # Leaky ReLU
            nn.Linear(8, 1),  # Output layer
            nn.Sigmoid()
        )
        self.output_layer.apply(init_weights)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.resnet_3d(X)
        X = self.output_layer(X)
        return X


class VideoMAE(torch.nn.Module):
    def __init__(self,
                 labels=None):
        super().__init__()

        if labels is None:
            labels = ['critical', 'non_critical']
        class_labels = sorted({item for item in labels})
        label2id = {label: i for i, label in enumerate(class_labels)}
        id2label = {i: label for label, i in label2id.items()}

        model_ckpt = "MCG-NJU/videomae-base"
        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_ckpt,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
            # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.model(X)
        return X


def l2_normalize(x, axis=2):
    return tf.nn.l2_normalize(x, axis=axis)


def get_bbox_info(predictions,
                  tracking_trajectories,
                  bboxes,
                  image_idx):
    """
    Updates tracking trajectories and bounding box information from model predictions.

    Args:
        predictions: The predictions object from the object detection model, expected to contain
                     bounding boxes (`boxes`), confidence scores (`conf`), class labels (`cls`),
                     and object IDs (`id`).
        tracking_trajectories (dict): A dictionary to store the trajectories of tracked objects.
                                      Keys are object IDs and values are deques containing
                                      the centroid coordinates of the objects.
        bboxes (list): A list to store bounding box information for each frame. Each entry is a
                       list with the format [bbox_coords, scores, classes, id_, image_idx].
        image_idx (int): The index of the current image/frame being processed.

    Returns:
        tuple:
            - tracking_trajectories (dict): Updated dictionary of object trajectories.
            - bboxes (list): Updated list of bounding box information.

    The function iterates through the bounding boxes in the predictions, extracts the coordinates,
    confidence scores, class labels, and IDs, and appends this information to the `bboxes` list.
    It also calculates the centroid of each bounding box and updates the `tracking_trajectories`
    dictionary with these centroids.

    Note:
        The function expects `predictions.boxes` to contain bounding box information, where each
        bounding box has `conf` (confidence scores), `cls` (class labels), `xyxy` (coordinates),
        and `id` (object IDs). If an ID is not `None` and is not already in `tracking_trajectories`,
        a new entry is created for it with a deque of max length 20.

        """
    for bbox in predictions.boxes:
        ## object detections
        for scores, classes, bbox_coords, id_ in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
            xmin = bbox_coords[0]
            ymin = bbox_coords[1]
            xmax = bbox_coords[2]
            ymax = bbox_coords[3]

            bboxes.append([bbox_coords, scores, classes, id_, image_idx])

            centroid_x = (xmin + xmax) / 2
            centroid_y = (ymin + ymax) / 2

            # Append centroid to tracking_points
            if id_ is not None and int(id_) not in tracking_trajectories:
                tracking_trajectories[int(id_)] = deque(maxlen=20)
            if id_ is not None:
                tracking_trajectories[int(id_)].append((centroid_x, centroid_y))

    return tracking_trajectories, bboxes


class Monocular2D:
    def __init__(self,
                 device):
        # Load a 2D model
        self.bbox2d_model = YOLO('yolov8n-seg.pt')  # load an official model
        # set model parameters
        self.bbox2d_model.overrides['conf'] = 0.5  # NMS confidence threshold
        self.bbox2d_model.overrides['iou'] = 0.45  # NMS IoU threshold
        self.bbox2d_model.overrides['agnostic_nms'] = False  # NMS class-agnostic
        self.bbox2d_model.overrides['max_det'] = 1000  # maximum number of detections per image
        self.bbox2d_model.overrides['classes'] = [0, 1, 2, 3, 4, 5, 6, 7]  # define classes
        self.yolo_classes = ['Pedestrian', 'Cyclist', 'Car', 'motorcycle', 'airplane', 'Van', 'train', 'Truck', 'boat']
        self.device = device

    def call(self,
             video: np.array):
        """
        Processes a video to track object trajectories and extract bounding box information.

        Args:
            video (np.array): A 4D NumPy array representing the video, where the shape is
                              (number_of_frames, height, width, channels).

        Returns:
            tuple:
                - bboxes (list): A list of lists containing bounding box information for each frame.
                                 Each sublist has the format [bbox_coords, scores, classes, id_, image_idx].
                - tracking_trajectories (dict_keys): Keys of the dictionary `tracking_trajectories`
                                                     representing the unique IDs of tracked objects.

        The function iterates through each frame of the video and applies the `bbox2d_model` to
        track objects in each frame. It collects bounding box information and updates the
        `tracking_trajectories` dictionary with the tracked objects' information.

        Note:
            The function expects `self.bbox2d_model.track` to return results that include
            `boxes`, `masks`, and `id`. If any of these are missing in the predictions,
            the frame is skipped.

            The `get_bbox_info` function is assumed to extract bounding box information and
            update `tracking_trajectories` and `bboxes`.

        """
        tracking_trajectories = {}
        bboxes = []  # bboxes is a list of list having format [bbox_coords, scores, classes, id_, image_idx]
        image_num = video.shape[0]
        for image_idx in range(image_num):
            image = video[image_idx, :, :, :]
            results = self.bbox2d_model.track(image,
                                              verbose=False,
                                              device=self.device,
                                              persist=image_idx != image_num - 1)

            for predictions in results:
                if predictions is None:
                    continue

                if predictions.boxes is None or predictions.masks is None or predictions.boxes.id is None:
                    continue

                tracking_trajectories, bboxes = get_bbox_info(predictions, tracking_trajectories, bboxes, image_idx)

        return bboxes, tracking_trajectories.keys()


class Monocular3D:
    def __init__(self, weights_path, input_shape=(224, 224, 3), base_arch='mobilenetv2'):
        self.weights_path = weights_path
        self.input_shape = input_shape
        self.base_arch = base_arch
        self.base_model = None
        self.model = None

        self.BIN, self.OVERLAP = 6, 0.1
        W = 1.
        ALPHA = 1.
        MAX_JIT = 3
        NORM_H, NORM_W = 224, 224
        VEHICLES = ['Car', 'Truck', 'Van', 'Tram']
        BATCH_SIZE = 8
        AUGMENTATION = False

        P2 = np.array(
            [[718.856, 0.0, 607.1928, 45.38225], [0.0, 718.856, 185.2157, -0.1130887], [0.0, 0.0, 1.0, 0.003779761]])
        dims_avg = {'Car': np.array([1.52131309, 1.64441358, 3.85728004]),
                    'Van': np.array([2.18560847, 1.91077601, 5.08042328]),
                    'Truck': np.array([3.07044968, 2.62877944, 11.17126338]),
                    'Pedestrian': np.array([1.75562272, 0.67027992, 0.87397566]),
                    'Person_sitting': np.array([1.28627907, 0.53976744, 0.96906977]),
                    'Cyclist': np.array([1.73456498, 0.58174006, 1.77485499]),
                    'Tram': np.array([3.56020305, 2.40172589, 18.60659898])}
        self.load_model()
        self.global_average = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
        self.flatten = tf.keras.layers.Flatten()

    def load_model(self):
        if self.base_arch == 'mobilenetv2':
            base_model = MobileNetV2(weights=None, include_top=False, input_shape=self.input_shape)
            x = base_model.output
            x = GlobalAveragePooling2D()(x)

            dimension = Dense(512)(x)
            dimension = LeakyReLU(alpha=0.1)(dimension)
            dimension = Dropout(0.2)(dimension)
            dimension = Dense(3)(dimension)
            dimension = LeakyReLU(alpha=0.1, name='dimension')(dimension)

            orientation = Dense(256)(x)
            orientation = LeakyReLU(alpha=0.1)(orientation)
            orientation = Dropout(0.2)(orientation)
            orientation = Dense(self.BIN * 2)(orientation)
            orientation = LeakyReLU(alpha=0.1)(orientation)
            orientation = Reshape((self.BIN, -1))(orientation)
            orientation = Lambda(l2_normalize, name='orientation')(orientation)

            confidence = Dense(256)(x)
            confidence = LeakyReLU(alpha=0.1)(confidence)
            confidence = Dropout(0.2)(confidence)
            confidence = Dense(self.BIN, activation='softmax', name='confidence')(confidence)

            model = Model(inputs=base_model.input, outputs=[dimension, orientation, confidence])
            model.load_weights(self.weights_path)

            self.model = model

            # Extract the base model
            self.base_model = Model(inputs=model.input, outputs=base_model.output)
        else:
            raise ValueError("Unsupported architecture")

    def call(self,
             img,):
        features = self.base_model(img, training=False)
        features = self.global_average(features)
        features = self.flatten(features)
        return features


class BinaryModel(tf.keras.Model):
    def __init__(self):
        super(BinaryModel, self).__init__()
        # Define your RNN layer
        self.rnn_for_extract_feature_per_object = tf.keras.layers.LSTM(64, return_sequences=False)
        # self.rnn_for_prediction = tf.keras.layers.LSTM(64, return_sequences=False)
        # Define the MLP layers
        self.global_average = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(16, activation='elu')
        self.dense2 = tf.keras.layers.Dense(4, activation='elu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, features_list, training=None, mask=None):
        final_feature_per_object = []
        for _, features in features_list.items():
            final_feature_per_object.append(self.rnn_for_extract_feature_per_object(tf.expand_dims(features, axis=0)))
        if len(final_feature_per_object) == 1:
            final_feature_per_object_reshape = tf.expand_dims(final_feature_per_object[0], axis=0)
        else:
            final_feature_per_object_reshape = tf.stack(final_feature_per_object, axis=1)
        final_feature = self.global_average(final_feature_per_object_reshape)
        x = self.dense1(final_feature)
        x = self.dense2(x)
        return self.output_layer(x)


class CriticalClassification(tf.keras.Model):
    def __init__(self,
                 mono3d_weights_path,
                 binary_model_weights_path,
                 device: str = 'cpu'):
        super(CriticalClassification, self).__init__()
        self.mono2d_model = Monocular2D(device=device)
        self.mono3d_model = Monocular3D(mono3d_weights_path)
        self.binary_model = BinaryModel.load_weights(binary_model_weights_path) \
            if os.path.exists(binary_model_weights_path) and 'binary' in binary_model_weights_path \
            else BinaryModel()

    def call(self, videos, training=None, mask=None):
        """
        Processes a batch of videos to detect bounding boxes, extract features, and predict critical events.

        Args:
            videos (np.array): A batch of videos represented as a 5D NumPy array with shape
                               (batch_size, channels, num_frames, height, width).
            training (optional): Unused argument for compatibility.
            mask (optional): Unused argument for compatibility.

        Returns:
            tf.Tensor: A tensor containing the predictions for each video in the batch, where each
                       prediction is the output of the binary model.
        """
        # Change video shape to (batch_size, num_frames, height, width, channels).
        videos = (np.array(videos).transpose((0, 2, 3, 4, 1)) * 255.0).astype(np.uint8)
        predictions = []

        for video in [videos[idx] for idx in range(videos.shape[0])]:
            # Predict bounding boxes using YOLO
            bboxes2d, id_list = self.mono2d_model.call(video=video)

            # Extract features using Monocular3D for each bounding box
            features = {}
            bboxes2d_dict = defaultdict(list)
            for item in bboxes2d:
                bbox_coords, scores, classes, id_, image_idx = item
                features[int(id_)] = []
                bboxes2d_dict[int(id_)].append(item)

            images_list = [video[idx] for idx in range(video.shape[0])]
            for id_, bboxes2d_of_id_ in bboxes2d_dict.items():
                patch_list = []
                for item in bboxes2d_of_id_:
                    bbox_coords, scores, classes, _, image_idx = item
                    padding = 0  # Set the padding value

                    # get part of the image from 2d bounding box and feed to 3d model
                    img = images_list[image_idx]
                    xmin = max(0, bbox_coords[0] - padding)
                    ymin = max(0, bbox_coords[1] - padding)
                    xmax = min(img.shape[1], bbox_coords[2] + padding)
                    ymax = min(img.shape[0], bbox_coords[3] + padding)

                    crop = img[int(ymin): int(ymax), int(xmin): int(xmax)]
                    patch = tf.convert_to_tensor(crop, dtype=tf.float32)
                    patch /= 255.0  # Normalize to [0,1]
                    patch = tf.image.resize(patch, (224, 224))
                    patch_list.append(patch)
                patch_concat = tf.stack(patch_list, axis=0)  # Final shape: (num_predicted_box, height, width, channels)
                features[int(id_)] = self.mono3d_model.call(patch_concat)

            # Predict critical event for a video using binary model if there are feature in image
            if len(features) == 0:
                predictions.append(tf.constant(0, dtype=tf.float32))
            else:
                predictions.append(tf.squeeze(self.binary_model.call(features)))

        return tf.stack(predictions, axis=0)
