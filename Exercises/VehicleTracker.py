from HeatMapper import HeatMapper
from car_classifier import car_classifier
from scipy.ndimage.measurements import label
from lesson_functions import search_windows, draw_boxes,slide_window2,draw_labeled_bboxes
#from lesson_functions import slide_window,search_windows,draw_boxes

import numpy as np
class VehicleTracker():
    def __init__(self, image_shape, window_sizes, y_start_stop,retrain=False):
        self.car_classifier = car_classifier(train=retrain)
        self.hmapper = HeatMapper(image_shape)
        self.windows = create_windows(image_shape, window_sizes, y_start_stop)

    def process(self, img,debug=False):
        if debug:
            print("debug process")
            process_img,boxes_img = process_pipeline(img, self.car_classifier, self.hmapper, self.windows,debug=True)
            return process_img, boxes_img
        else:
            process_img = process_pipeline(img, self.car_classifier, self.hmapper, self.windows,debug=False)
            return process_img


def process_pipeline(img, car_classifier, heatmapper, windows, debug=False):

    norm_img = img.astype(np.float32)/255.0
    box_list = search_windows(img, windows, car_classifier.clf,
                              car_classifier.feature_scaler,
                              car_classifier.color_space,
                              car_classifier.spatial_size,
                              car_classifier.hist_bins,
                              (0,256),
                              car_classifier.orient,
                              car_classifier.pix_per_cell,
                              car_classifier.cell_per_block,
                              car_classifier.hog_channel,
                              car_classifier.spatial_feat,
                              car_classifier.hist_feat,
                              car_classifier.hog_feat)
    if debug:
        boxes_img = np.copy(img)
        boxes_img = draw_boxes(img,box_list)

    # creat heatmap using box_list
    heatmap = heatmapper.compute_heatmap(box_list)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    if debug:
        print("debug process pipeline")
        return draw_img, boxes_img
    else:
        return draw_img


def create_windows(img_shape, window_sizes, y_start_stop):
    windows = []
    for i in range(len(window_sizes)):
        yval = (np.array(y_start_stop[i]) * (img_shape[0]) / 100.0)
        # print(yval)
        windows += slide_window2(
            img_shape, x_start_stop=[None, None], y_start_stop=[int(yval[0]), int(yval[1])],
            xy_window=(window_sizes[i], window_sizes[i]), xy_overlap=(0.8, 0.8)
        )
    return windows
