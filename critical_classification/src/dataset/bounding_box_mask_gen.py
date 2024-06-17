# Get video from input path, and relevant info from metadata, including when the sequence of images are considered
# critical driving scenario (ground truth)
# Get 2D object detection prediction for all the image in the video
# Use that as the input to critical driving scenario model (nn.Module)
# Train such a binary model
# Save the model after training
# do inference on some videos

from torchvision import transforms
from PIL import Image
import argparse
import os, sys
import cv2
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from critical_classification.src.models_for_project import YOLOv1
from critical_classification.src.dataset.bounding_box_representation import draw_gaussian, draw_rectangle

# All BDD100K (dataset) classes and the corresponding class colors for drawing
# the bounding boxes
category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                 "truck", "train", "other person", "bus", "car", "rider",
                 "motorcycle", "bicycle", "trailer"]
interest_list = ["other vehicle", "pedestrian", "truck", "train", "other person", "bus", "car", "rider",
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
ap.add_argument("-i", "--dataset_path",
                help="path to dataset", default='critical_classification/dashcam_video/original_video')
args = ap.parse_args()


def bounding_box_mask_gen_single_video(input_path,
                                       transform,
                                       device,
                                       model
                                       ):
    print("Loading input video file")
    print("...")
    vs = cv2.VideoCapture(input_path)
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Defining the output video file
    print(f'output bounding box mask save at'
          f'{os.path.dirname(os.path.dirname(input_path))}/bounding_box_mask_video/'
          f'{os.path.basename(input_path)[:-4]}_mask.mp4'
          )
    out = cv2.VideoWriter(f'{os.path.dirname(os.path.dirname(input_path))}/bounding_box_mask_video/'
                          f'{os.path.basename(input_path)[:-4]}_mask.mp4',
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          30,
                          (frame_width, frame_height))

    # Used to scale the bounding box predictions to the original input frame
    # (448 is the dimension of the input image for the model)
    ratio_x = frame_width / 448
    ratio_y = frame_height / 448

    amount_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))  # Amount of frames

    for _ in tqdm(range(amount_frames)):
        grabbed, frame = vs.read()

        # Create a mask
        mask = np.zeros_like(frame, dtype=np.float32)
        mask_single_channel = np.zeros_like(frame[:, :, 0], dtype=np.float32)

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

                # Extracts the box confidence score, the box coordinates and class
                confidence_score = output[0, cell_h, cell_w, best_box * 5]
                center_box = output[0, cell_h, cell_w, best_box * 5 + 1:best_box * 5 + 5]
                best_class = corr_class[cell_h, cell_w]

                # Checks if the confidence score is above the specified threshold
                if confidence_score <= float(args.threshold) and best_class not in interest_list:
                    continue

                # Transforms the box coordinates into pixel coordinates
                centre_x = (center_box[0] * 32 + 32 * cell_w)
                centre_y = (center_box[1] * 32 + 32 * cell_h)
                width = center_box[2] * 448
                height = center_box[3] * 448

                mask_single_channel = draw_gaussian(
                    mask_single_channel,
                    blob_center=[int(centre_x * ratio_x), int(centre_y * ratio_y)],
                    blob_width_and_height=[int(width), int(height)]
                )
                mask = cv2.merge([mask_single_channel, mask_single_channel, mask_single_channel])

                mask = draw_rectangle(mask,
                                      centre=[centre_x, centre_y],
                                      width_and_height=[width, height],
                                      ratio=[ratio_x, ratio_y])

        out.write(np.uint8(mask))


def bounding_box_mask_gen(dataset_path: str):
    print("##### YOLO OBJECT DETECTION FOR VIDEOS #####")
    print("Loading the model")
    print("...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda')
    model = YOLOv1(int(args.split_size), int(args.num_boxes), int(args.num_classes)).to(device)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Amount of YOLO parameters: " + str(num_param))
    print("...")
    print("Loading model weights")
    print("...")
    weights = torch.load(args.weights, map_location='cuda')
    model.load_state_dict(weights["state_dict"])
    model.eval()

    # Transform is applied to the input frames
    # It resizes the image and converts it into a tensor
    transform = transforms.Compose([
        transforms.Resize((448, 448), Image.NEAREST),
        transforms.ToTensor(),
    ])

    for filename in os.listdir(dataset_path):
        # Construct the full filepath
        file_path = os.path.join(dataset_path, filename)

        bounding_box_mask_gen_single_video(input_path=file_path,
                                           model=model,
                                           transform=transform,
                                           device=device)


if __name__ == '__main__':
    bounding_box_mask_gen(dataset_path=args.dataset_path)
