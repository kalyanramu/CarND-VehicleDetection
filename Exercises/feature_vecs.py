from skimage.feature import hog
import cv2
import numpy as np
from lesson_functions import *



def get_hog_features(img, num_orient_bins=9, pix_per_cell=8, cell_per_block=2,feature_vec=True,vis=False):
    #print("len of image",img.shape)
    # HOG features from skimage takes only 2D image, if color image is passed it throws an error

    hog_features = []
    if len(img.shape) > 2:
        num_channels = img.shape[-1]
    else:
        num_channels = 1
    for i in range(num_channels):
        channel_hog_feature = hog(img[:,:,i], orientations=num_orient_bins, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), visualise=False, feature_vector=feature_vec)
        hog_features.append(channel_hog_feature)

    return np.concatenate(hog_features)

# Define a function to compute binned color features
def get_binspatial_features(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    binspatial_features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return binspatial_features

# Define a function to compute color histogram features
def get_colorhist_features(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_multiple_features(img,features_list=None,feature_channel=None):
    if features_list == None:
        features_list =['hog','binspatial','colorhist']

    img_features = []
    for feature in features_list:
        if feature == 'hog':
            #print("HOG:",get_hog_features(img).shape)
            img_features.append(get_hog_features(img))
        if feature == 'binspatial':
            #print("BIN SPATIAL:",get_binspatial_features(img).shape)
            img_features.append(get_binspatial_features(img))
        if feature == 'colorhist':
            #print("COLOR:",get_colorhist_features(img).shape)
            img_features.append(get_colorhist_features(img))

    all_features = np.concatenate(img_features)
    #print("ALL FEATURES",all_features.shape)
    return(all_features)