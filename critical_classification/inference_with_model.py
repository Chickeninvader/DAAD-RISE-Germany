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
import time
import cv2
import torch

# All BDD100K (dataset) classes and the corresponding class colors for drawing
# the bounding boxes
category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                 "truck", "train", "other person", "bus", "car", "rider",
                 "motorcycle", "bicycle", "trailer"]
category_color = [(255, 255, 0), (0, 0, 255), (255, 128, 0), (0, 255, 255), (255, 0, 255),
                  (128, 255, 0), (0, 255, 128), (255, 0, 127), (0, 0, 255), (0, 0, 255),
                  (127, 0, 255), (0, 0, 255), (128, 128, 128)]

# Argparse to apply YOLO algorithm to a video file from the console
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to the modell weights")
ap.add_argument("-i", "--input", required=True, help="path to your input video")
ap.add_argument("-o", "--output", required=True, help="path to your output video")
args = ap.parse_args()

def main():

    # Transform is applied to the input frames
    # It resizes the image and converts it into a tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224), Image.NEAREST),
        transforms.ToTensor(),
    ])

    print("Loading input video file")
    print("...")
    vs = cv2.VideoCapture(args.input)
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Defining the output video file
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                          (frame_width, frame_height))

    idx = 1  # Used to track how many frames have been already processed
    sum_fps = 0  # Used to track the average FPS at the end
    amount_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))  # Amount of frames

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break

        idx += 1  # Frame index
        img = Image.fromarray(frame)
        img_tensor = transform(img).unsqueeze(0).to(device)
        img = cv2.UMat(frame)

        with torch.no_grad():
            start_time = time.time()
            output = model(img_tensor)  # Makes a prediction on the input frame
            curr_fps = int(1.0 / (time.time() - start_time))  # Prediction FPS
            sum_fps += curr_fps

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
                    centre_x = center_box[0] * 32 + 32 * cell_w
                    centre_y = center_box[1] * 32 + 32 * cell_h
                    width = center_box[2] * 448
                    height = center_box[3] * 448

                    # Calculates the corner values of the bounding box
                    x1 = int((centre_x - width / 2) * ratio_x)
                    y1 = int((centre_y - height / 2) * ratio_y)
                    x2 = int((centre_x + width / 2) * ratio_x)
                    y2 = int((centre_y + height / 2) * ratio_y)

                    # Draws the bounding box with the corresponding class color
                    # around the object
                    cv2.rectangle(img, (x1, y1), (x2, y2), category_color[best_class], 1)
                    # Generates the background for the text, painted in the corresponding
                    # class color and the text with the class label including the
                    # confidence score
                    labelsize = cv2.getTextSize(category_list[best_class],
                                                cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - 20), (x1 + labelsize[0][0] + 45, y1),
                                  category_color[best_class], -1)
                    cv2.putText(img, category_list[best_class] + " " +
                                str(int(confidence_score.item() * 100)) + "%", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    # Generates a small window in the top left corner which
                    # displays the current FPS for the prediction
                    # cv2.putText(img, str(curr_fps) + "FPS", (25, 30),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                    img = manual_detection(corner=[x1, y1, x2, y2],
                                           ratio_x=ratio_x,
                                           ratio_y=ratio_y,
                                           img=img)

        out.write(img)  # Stores the frame with the predictions on a new mp4 file
    print("Average FPS was: " + str(int(sum_fps / amount_frames)))
    print("")


if __name__ == '__main__':
    main()
