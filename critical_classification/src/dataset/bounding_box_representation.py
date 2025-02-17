import numpy as np
import cv2

# ref:
# https://stackoverflow.com/questions/76723027/how-to-draw-2d-gaussian-blob-on-an-opencv-image/76724003#76724003


def draw_rectangle(heat_map,
                   centre,
                   width_and_height,
                   ratio,):
    centre_x, centre_y = centre
    width, height = width_and_height
    ratio_x, ratio_y = ratio

    # Calculates the corner values of the bounding box
    x1 = int((centre_x - width / 2) * ratio_x)
    y1 = int((centre_y - height / 2) * ratio_y)
    x2 = int((centre_x + width / 2) * ratio_x)
    y2 = int((centre_y + height / 2) * ratio_y)

    return cv2.rectangle(heat_map, (x1, y1), (x2, y2), (255, 255, 255), 2)


def draw_gaussian(heatmap,
                  blob_center,
                  blob_width_and_height):
    # suppress numpy printing in scientific notation
    np.set_printoptions(suppress=True)

    blob_1_center_x, blob_1_center_y = blob_center
    blob_1_width, blob_1_height = int(blob_width_and_height[0] * 1.2), int(blob_width_and_height[1] * 1.2)

    if blob_1_width % 2 == 0:
        blob_1_width += 1

    if blob_1_height % 2 == 0:
        blob_1_height += 1

    gaussian_blob_1 = make_gaussian_blob(blob_1_width, blob_1_height)
    heatmap = add_gaussian_blob_to_heatmap(gaussian_blob_1, blob_1_center_x, blob_1_center_y, heatmap)

    return heatmap


def make_gaussian_blob(blob_width, blob_height):
    assert blob_height % 2 == 1 and blob_width % 2 == 1, (f'in make_gaussian_blob, '
                                                          f'blob_height and blob_width must be odd numbers !!')

    # Create a 2D Gaussian blob
    # +-2.5 was derived from experimentation
    x, y = np.meshgrid(np.linspace(-2.5, 2.5, blob_width), np.linspace(-2.5, 2.5, blob_height))

    gaussian_blob = np.exp(-0.5 * (x ** 2 + y ** 2))

    # scale up the gaussian blob from the 0.0 to 1.0 range to the 0 to 255 range
    gaussian_blob = gaussian_blob * 255.0
    gaussian_blob = np.clip(gaussian_blob, a_min=0.0, a_max=255.0)
    gaussian_blob = np.rint(gaussian_blob).astype(np.uint8)

    return gaussian_blob


# end function

def add_gaussian_blob_to_heatmap(gaussian_blob, blob_center_x, blob_center_y, heatmap):

    blob_height, blob_width = gaussian_blob.shape[0:2]
    blob_left_edge_loc = round(blob_center_x - ((blob_width - 1) * 0.5))
    blob_right_edge_loc = round(blob_center_x + ((blob_width - 1) * 0.5))
    gaussian_left_edge = 0
    gaussian_right_edge = gaussian_blob.shape[1] - 1
    if blob_left_edge_loc < 0:
        gaussian_left_edge = -blob_left_edge_loc
        blob_left_edge_loc = 0
    if blob_right_edge_loc >= heatmap.shape[1]:
        gaussian_right_edge = gaussian_blob.shape[1] - 1 - (blob_right_edge_loc - (heatmap.shape[1] - 1))
        blob_right_edge_loc = heatmap.shape[1] - 1

    blob_top_edge_loc = round(blob_center_y - ((blob_height - 1) * 0.5))
    blob_bottom_edge_loc = round(blob_center_y + ((blob_height - 1) * 0.5))
    gaussian_top_edge = 0
    gaussian_bottom_edge = gaussian_blob.shape[0] - 1
    if blob_top_edge_loc < 0:
        gaussian_top_edge = -blob_top_edge_loc
        blob_top_edge_loc = 0
    if blob_bottom_edge_loc >= heatmap.shape[0]:
        gaussian_bottom_edge = gaussian_blob.shape[0] - 1 - (blob_bottom_edge_loc - (heatmap.shape[0] - 1))
        blob_bottom_edge_loc = heatmap.shape[0] - 1

    heatmap = heatmap.astype(np.uint16)
    gaussian_blob = gaussian_blob.astype(np.uint16)

    heatmap[blob_top_edge_loc:blob_bottom_edge_loc, blob_left_edge_loc:blob_right_edge_loc] += gaussian_blob[
                                                                                               gaussian_top_edge:gaussian_bottom_edge,
                                                                                               gaussian_left_edge:gaussian_right_edge]

    heatmap = np.where(heatmap > 255, 255, heatmap).astype(np.uint8)

    return heatmap