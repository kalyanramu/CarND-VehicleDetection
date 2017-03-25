#from car_classifier import car_classifier
#clf = car_classifier(train=False)

# from HeatMapper import HeatMapper
# hm = HeatMapper((10,10))

import matplotlib.image as mpimg
from VehicleTracker import VehicleTracker
test_img = mpimg.imread('./test_images/test3.jpg')
window_sizes = [64,96,128]
y_start_stop = [[50, 70],[50, 80],[50, 90]]
img_shape = test_img.shape

tracker = VehicleTracker(img_shape,window_sizes,y_start_stop)