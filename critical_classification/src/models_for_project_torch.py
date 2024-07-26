import os
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision
from transformers import VideoMAEForVideoClassification
from torch.autograd import Variable
import torchvision.models as models

sys.path.append(os.getcwd())

from critical_classification.config import Config


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


class YOLOv1_image_binary(nn.Module):
    """
    This class contains the YOLOv1 model. It consists of 24 convolutional and
    2 fully-connected layers which divide the input image into a
    (split_size x split_size) grid and predict num_boxes bounding boxes per grid
    cell.
    """

    def __init__(self, split_size, num_boxes, num_classes, device=torch.device('cpu')):
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

        super(YOLOv1_image_binary, self).__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.pretrain_base_model_path = 'critical_classification/save_models/YOLO_bdd100k.pt'
        self.category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                              "truck", "train", "other person", "bus", "car", "rider",
                              "motorcycle", "bicycle", "trailer"]
        self.device = device

        self.base_model = YOLOv1(self.split_size, self.num_boxes, self.num_classes).to(self.device)
        weights = torch.load(self.pretrain_base_model_path, map_location=self.device)

        # Freeze base layer
        self.base_model.load_state_dict(weights["state_dict"])
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.flatten = nn.Flatten()

        # Remove last layer (fc layer) and Add extra layer returning binary output only
        self.base_model.fc = nn.Sequential(
            nn.Linear(int(1024 * self.split_size * self.split_size / 4), 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # Add new fully-connected layer for binary output
        self.binary_fc = nn.Sequential(
            nn.Linear(4096, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
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

        x = self.base_model.darkNet(x)
        x = self.flatten(x)
        x = self.base_model.fc(x)
        x = self.binary_fc(x)

        return x


class YOLOv1_video_binary(nn.Module):
    """
        This class contains the YOLOv1 model. It consists of 24 convolutional and
        2 fully-connected layers which divide the input image into a
        (split_size x split_size) grid and predict num_boxes bounding boxes per grid
        cell.
        """

    def __init__(self, split_size, num_boxes, num_classes, device, config: Config):
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

        super(YOLOv1_video_binary, self).__init__()
        self.split_size = split_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.config = config
        self.pretrain_base_model_path = 'critical_classification/save_models/YOLO_bdd100k.pt'
        self.category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                              "truck", "train", "other person", "bus", "car", "rider",
                              "motorcycle", "bicycle", "trailer"]
        self.device = device

        self.base_model = YOLOv1(self.split_size, self.num_boxes, self.num_classes).to(self.device)
        weights = torch.load(self.pretrain_base_model_path, map_location=self.device)

        # Freeze darknet layer
        self.base_model.load_state_dict(weights["state_dict"])
        for param in self.base_model.darkNet.parameters():
            param.requires_grad = False

        # Add new LSTM layer for binary output
        self.hidden_size = 256  # hidden size of lstm
        self.num_layers = 4  # number of LSTM layers stacked

        if 'no_fc' in config.additional_config:
            self.LSTM = torch.nn.LSTM(input_size=1024 * split_size * split_size,
                                      hidden_size=self.hidden_size,
                                      num_layers=self.num_layers)
        else:
            self.LSTM = torch.nn.LSTM(input_size=split_size * split_size * (num_classes + num_boxes * 5),
                                      hidden_size=self.hidden_size,
                                      num_layers=self.num_layers)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_layers, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forwards the input tensor through the model to produce the predictions.

        Parameters:
            x (tensor): A tensor of shape (num_frames, 3, 448, 448) which represents
            a batch of input images.

        Returns:
            x (tensor): A tensor of shape () which contains the predicted critical value
        """
        # Since there is alway 1 video process only, the 'batch_size' is 1
        h_0 = Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).to(self.device)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).to(self.device)  # internal state

        x = self.base_model.darkNet(x)
        x = torch.flatten(x, start_dim=1)

        if 'no_fc' not in self.config.additional_config:
            x = self.base_model.fc(x)

        x = x.unsqueeze(1)

        output, (hn, cn) = self.LSTM(x, (h_0.detach(), c_0.detach()))
        hn_last = hn.reshape(-1)
        final_output = self.fc(hn_last)

        return final_output

    # def infer_from_video(self, x):
    #     """
    #     Forwards the input tensor through the model to produce the predictions.
    #
    #     Parameters:
    #         x (tensor): A tensor of shape (num_frames, 3, 448, 448) which represents
    #         multiple batch of input images.
    #
    #     Returns:
    #         final_output (tensor): A tensor of shape () which contains the predicted critical value
    #     """
    #     # Since there is alway 1 video process only, the 'batch_size' is 1
    #     h_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)).to(self.device)  # hidden state
    #     c_0 = Variable(torch.zeros(self.num_layers, self.hidden_size)).to(self.device)  # internal state
    #
    #     x = self.base_model.darkNet(x)
    #     x = torch.flatten(x, start_dim=1)
    #     x = self.base_model.fc(x)
    #     x = x.unsqueeze(1)
    #     print(f'Yolo feature shape: {x.shape}')
    #
    #     h_n, c_n = h_0, c_0
    #     hidden_states = []
    #
    #     for t in range(x.size(0)):
    #         h_n, c_n = self.LSTM(x[t, :, :], (h_n, c_n))
    #         hidden_states.append(h_n)
    #
    #     print(f'hidden state at each step shape: {h_n.shape}')
    #     hidden_states = torch.stack(hidden_states, dim=1)
    #     print(f'hidden state shape is {hidden_states.shape}')
    #     final_output_list = []
    #
    #     # The original model is train with num_frames = 15
    #     for i in range(x.size(0) - 15):
    #         hn_last = hidden_states[i + 15, :, :].reshape(-1)
    #         final_output_list.append(self.fc(hn_last))
    #     print(f'number of prediction: {len(final_output_list)}')
    #     return final_output_list


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class ResNet3D(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

        self.resnet_3d = torchvision.models.video.r3d_18(pretrained=True)
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


class VideoSwinTransformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # pass dropout 0.1
        if 'train_from_scratch' in config.additional_config:
            # train from scratch
            self.model = models.video.swin3d_b(dropout=0.1)
        else:
            self.model = models.video.swin3d_b(weights=models.video.Swin3D_B_Weights.KINETICS400_V1)

        # Suppose the final layer is named 'head'
        num_features = self.model.head.in_features

        # Replace the final layer
        self.model.head = torch.nn.Linear(num_features, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1, 3, 4))
        return self.sigmoid(self.model(x))


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        # You can define dummy layers here if needed, e.g.:
        self.dummy_layer = nn.Identity()
        self.dummy_linear = nn.Linear(1, 1)

    def forward(self, x):
        # Dummy forward pass: just return the input tensor
        return self.dummy_linear(x)
