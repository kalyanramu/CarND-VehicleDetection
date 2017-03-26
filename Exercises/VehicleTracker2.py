from HeatMapper import HeatMapper
from car_classifier import car_classifier
from scipy.ndimage.measurements import label
from lesson_functions import get_hog_features, bin_spatial,color_hist,draw_labeled_bboxes, convert_color
import cv2
import numpy as np
#from lesson_functions import slide_window,search_windows,draw_boxes

import numpy as np
class VehicleTracker2():
    def __init__(self,image_shape,heatmap_threshold=1,retrain=False):
        self.car_classifier = car_classifier(train=retrain)
        self.heatmapper = HeatMapper(image_shape)
        self.heatmap_threshold = heatmap_threshold
        self.scales = [1]

    def debug_process(self,img,threshold=1,debug=False):
        ystart = 400
        ystop = 656
        scale = 1.5
        orient = 9
        pix_per_cell= 8
        cell_per_block= 2
        spatial_size= (32, 32)
        hist_bins= 32
        svc = self.car_classifier.clf
        X_scaler = self.car_classifier.feature_scaler
        init_img, box_list = find_cars(img, 
                                    ystart, ystop, scale, svc, X_scaler, 
                                    orient, pix_per_cell, cell_per_block, 
                                    spatial_size, hist_bins)
        heatmap = self.heatmapper.compute_heatmapN(box_list,threshold)
        labels = label(heatmap)
        final_img = draw_labeled_bboxes(np.copy(img), labels)

        return final_img,init_img,box_list,heatmap

    def process(self,img):
        ystart = 400
        ystop1 = 656
        ystop2 = 500
        scale1 = 1.5
        scale2 = 0.75
        orient = 9
        pix_per_cell= 8
        cell_per_block= 2
        spatial_size= (32, 32)
        hist_bins= 32
        threshold = self.heatmap_threshold
        svc = self.car_classifier.clf
        X_scaler = self.car_classifier.feature_scaler
        init_img, box_list1 = find_cars(img, 
                                    ystart, ystop1, scale1, svc, X_scaler, 
                                    orient, pix_per_cell, cell_per_block, 
                                    spatial_size, hist_bins)
        init_img, box_list2 = find_cars(img, 
                                    ystart, ystop2, scale2, svc, X_scaler, 
                                    orient, pix_per_cell, cell_per_block, 
                                    spatial_size, hist_bins)
        box_list = box_list1 + box_list2
        heatmap = self.heatmapper.compute_heatmapN(box_list,threshold)
        labels = label(heatmap)
        final_img = draw_labeled_bboxes(np.copy(img), labels)

        return final_img

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255.0
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    car_boxes=[]
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1: # and svc.decision_function(test_features) > 0.3: #if car found
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)

                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                startx = xbox_left 
                starty = ytop_draw+ystart
                endx = xbox_left+win_draw
                endy = ytop_draw+win_draw+ystart
                car_boxes.append(((startx,starty),(endx,endy)))
    return draw_img, car_boxes

