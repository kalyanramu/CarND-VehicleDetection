#references
#dumping and loading classifiers : http://scikit-learn.org/stable/modules/model_persistence.html

from lesson_functions import extract_features
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

import glob
import numpy as np

class car_classifier:
	def __init__(self,train=False,verbose=False):
		#Load from file as training from scratch takes lot of time
		self.clf = None
		self.feature_scaler = None

		#Read number of histbins from config file
		self.color_space = 'YCrCb' #'RGB'
		self.spatial_size = (32,32)
		self.hist_bins = 32
		self.orient = 9
		self.pix_per_cell =8
		self.cell_per_block = 2
		self.hog_channel = 'ALL'
		self.spatial_feat= True
		self.hist_feat = True
		self.hog_feat = True
		
		#Load classifier and feature scaler from pre-saved files
		if verbose:
			print("Train:",train)
		if train == False:
			try:
				self.clf = joblib.load("./dependencies/car_classifier.pkl")
				self.feature_scaler = joblib.load("./dependencies/car_feature_scaler.pkl")
				if verbose:
					print("clf: ",self.clf)
					print("scaler:",self.feature_scaler)
			except Exception as e: 
					print("exception thrown",str(e))


		#Train from scratch if classifier pkl file is not found
		if self.clf == None:
			print("Retraining Car Classifier")
			self.train_classifier()

	def train_classifier(self):
		vehicles = glob.glob('./Data/vehicles/**/*.png',recursive=True)
		non_vehicles = glob.glob('./Data/non-vehicles/**/*.png',recursive=True)
		print("Number of images in vehicle dataset:",len(vehicles))
		print("Number of images in non-vehicle dataset:",len(non_vehicles))


		#Read training data
		vehicle_features 			= extract_features(vehicles, 
																				color_space=self.color_space, spatial_size=self.spatial_size,
                        								hist_bins=self.hist_bins, orient=self.orient, 
                        								pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, 
                        								hog_channel= self.hog_channel,
                        								spatial_feat=self.spatial_feat, hist_feat=self.hist_feat, 
                        								hog_feat=self.hog_feat)
		non_vehicle_features 	= extract_features(non_vehicles, 
																				color_space=self.color_space, spatial_size=self.spatial_size,
                        								hist_bins=self.hist_bins, orient=self.orient, 
                        								pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block, 
                        								hog_channel= self.hog_channel,
                        								spatial_feat=self.spatial_feat, hist_feat=self.hist_feat, 
                        								hog_feat=self.hog_feat)

		#Define labels vectors bases on features
		feature_labels = np.hstack((np.ones(len(vehicles)),np.zeros(len(non_vehicles))))
		X_train_initial = np.vstack((vehicle_features,non_vehicle_features))
		y_train_initial = feature_labels
		
		#Normalize/scale features
		feature_scaler = StandardScaler()
		X_train_scaled = feature_scaler.fit_transform(X_train_initial)

		X_train_final = X_train_scaled
		y_train_final = y_train_initial

		#Train the classifier
		from sklearn.model_selection import train_test_split
		rand_state = np.random.randint(0, 100)
		X_train,X_test,y_train,y_test = train_test_split(X_train_final,y_train_final,test_size=0.2,
                                                 random_state = rand_state)

		from sklearn.svm import LinearSVC
		clf = LinearSVC()
		clf.fit(X_train,y_train)
		#Print accuracy of SVC
		print("End of Training Classifier")
		print('Test Accuracy of SVC = ', clf.score(X_test, y_test))

		
		#Save the classifier
		self.clf = clf
		self.feature_scaler = feature_scaler
		print("Saving Classifier and Feature Scaler to file")
		joblib.dump(clf,"./dependencies/car_classifier.pkl")
		joblib.dump(feature_scaler,"./dependencies/car_feature_scaler.pkl")

		return 