# ------------------------------------------------------------------------
# Visualization code for InstanceFormer
# ------------------------------------------------------------------------

import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize(img, cmap='binary'):
    plt.imshow(img, cmap=cmap)
    plt.show(block=True)


def visualize_bbox(image, bbox):
    # image = copy.deepcopy(image)
    image = np.ascontiguousarray(image)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    visualize(image)


def visualize_segmentation(img, mask, seq='title'):
    plt.subplot(2, 1, 1)
    plt.title(seq)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.imshow(mask)
    plt.show(block=True)


def visualize_polygon(image, pts):
    # image = copy.deepcopy(image)
    isClosed = True
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    for p in pts:
        p = np.array(p, np.int32).reshape((-1, 2))[:, None, :]
        cv2.polylines(image, [p], isClosed, color, thickness)
    visualize(image)