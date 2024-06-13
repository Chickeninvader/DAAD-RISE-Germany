# Get video from input path, and relevant info from metadata, including when the sequence of images are considered
# critical driving scenario (ground truth)
# Get 2D object detection prediction for all the image in the video
# Use that as the input to critical driving scenario model (nn.Module)
# Train such a binary model
# Save the model after training
# do inference on some videos

from torchvision import transforms
from models import YOLOv1
from PIL import Image
import argparse
import time
import os
import cv2
import torch
import numpy as np
import matplotlib.path as mpltPath
from tqdm import tqdm

# All BDD100K (dataset) classes and the corresponding class colors for drawing
# the bounding boxes
category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                 "truck", "train", "other person", "bus", "car", "rider",
                 "motorcycle", "bicycle", "trailer"]
category_color = [(255, 255, 0), (255, 0, 0), (255, 128, 0), (0, 255, 255), (255, 0, 255),
                  (128, 255, 0), (0, 255, 128), (255, 0, 127), (0, 255, 0), (0, 0, 255),
                  (127, 0, 255), (0, 128, 255), (128, 128, 128)]

# Argparse to apply YOLO algorithm to a video file from the console
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to the model weights")
ap.add_argument("-t", "--threshold", default=0.5,
                help="threshold for the confidence score of the bounding box prediction")
ap.add_argument("-ss", "--split_size", default=14,
                help="split size of the grid which is applied to the image")
ap.add_argument("-nb", "--num_boxes", default=2,
                help="number of bounding boxes which are being predicted")
ap.add_argument("-nc", "--num_classes", default=13,
                help="number of classes which are being predicted")
ap.add_argument("-i", "--input", required=True, help="path to your input video")
ap.add_argument("-o", "--output", required=True, help="path to your output video")
args = ap.parse_args()


def manual_detection(centre_x,
                     centre_y,
                     ratio_x,
                     ratio_y,
                     img):
    # Define region for critical driving scenario:
    # List of points, each point being a tuple (x, y)
    point_list = [(200, 300),
                  (50, 447),
                  (447 - 50, 447),
                  (447 - 200, 300),
                  ]

    # Convert the list to a NumPy array with the expected format
    points = np.array(point_list, dtype=np.int32)
    points[:, 0] = points[:, 0] * ratio_x
    points[:, 1] = points[:, 1] * ratio_y

    # display text when there is critical driving scenario
    path = mpltPath.Path(vertices=points)
    if path.contains_points([(int(centre_x * ratio_x), int(centre_y * ratio_y))]):
        cv2.putText(img,
                    text='Critical driving scenario',
                    org=(int(100 * ratio_x), int(100 * ratio_y)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=5,
                    color=(0, 0, 0),
                    thickness=2)
        # Draw lines connecting the points
        point_list = [(int(item[0] * ratio_x), int(item[1] * ratio_y)) for item in point_list]

        for point_idx in range(len(point_list) - 1):
            cv2.line(img, point_list[point_idx], point_list[point_idx + 1], color=(0, 0, 0), thickness=3)
        cv2.line(img, point_list[-1], point_list[0], color=(0, 0, 0), thickness=3)

    return img


def neural_network_detection():
    pass


def main():
    print("")
    print("##### YOLO OBJECT DETECTION FOR VIDEOS #####")
    print("")
    print("Loading the model")
    print("...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cpu')
    model = YOLOv1(int(args.split_size), int(args.num_boxes), int(args.num_classes)).to(device)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Amount of YOLO parameters: " + str(num_param))
    print("...")
    print("Loading model weights")
    print("...")
    weights = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(weights["state_dict"])
    model.eval()

    # Transform is applied to the input frames
    # It resizes the image and converts it into a tensor
    transform = transforms.Compose([
        transforms.Resize((448, 448), Image.NEAREST),
        transforms.ToTensor(),
    ])

    print("Loading input video file")
    print("...")
    vs = cv2.VideoCapture(args.input)
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Defining the output video file
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                          (frame_width, frame_height), isColor=False)

    # Used to scale the bounding box predictions to the original input frame
    # (448 is the dimension of the input image for the model)
    ratio_x = frame_width / 448
    ratio_y = frame_height / 448

    amount_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))  # Amount of frames

    for _ in tqdm(range(amount_frames)):
        grabbed, frame = vs.read()

        # Create a mask
        mask = np.zeros_like(frame, dtype=np.float32)

        img = Image.fromarray(frame)
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)  # Makes a prediction on the input frame

        # Extracts the class index with the highest confidence scores
        corr_class = torch.argmax(output[0, :, :, 10:23], dim=2)

        for cell_h in range(output.shape[1]):
            for cell_w in range(output.shape[2]):
                # Determines the best bounding box prediction
                best_box = 0
                max_conf = 0
                for box in range(int(args.num_boxes)):
                    if output[0, cell_h, cell_w, box * 5] > max_conf:
                        best_box = box
                        max_conf = output[0, cell_h, cell_w, box * 5]

                # Checks if the confidence score is above the specified threshold
                if output[0, cell_h, cell_w, best_box * 5] >= float(args.threshold):
                    # Extracts the box confidence score, the box coordinates and class
                    confidence_score = output[0, cell_h, cell_w, best_box * 5]
                    center_box = output[0, cell_h, cell_w, best_box * 5 + 1:best_box * 5 + 5]
                    best_class = corr_class[cell_h, cell_w]

                    # Transforms the box coordinates into pixel coordinates
                    centre_x = (center_box[0] * 32 + 32 * cell_w) * ratio_x
                    centre_y = (center_box[1] * 32 + 32 * cell_h) * ratio_y
                    width = center_box[2] * 448
                    height = center_box[3] * 448

                    # Calculates the corner values of the bounding box
                    x1 = int((centre_x - width / 2) * ratio_x)
                    y1 = int((centre_y - height / 2) * ratio_y)
                    x2 = int((centre_x + width / 2) * ratio_x)
                    y2 = int((centre_y + height / 2) * ratio_y)

                    cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), 2)
        out.write(np.uint8(mask))

if __name__ == '__main__':
    main()
