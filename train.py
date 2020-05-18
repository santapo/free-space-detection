# -*- coding: utf-8 -*-

import torch
import importlib
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

from shapely.geometry import Polygon, MultiPolygon

from concurrent.futures import ThreadPoolExecutor, as_completed
from google.colab.patches import cv2_imshow

import numpy as np
import cv2

import sys
import os
import time

### Colors for visualization
# Ego, left, right
COLORS = [(255,0,0), (0,255,0), (0,0,255)]

# Road name map
ROAD_MAP = ['Residential', 'Highway', 'City Street', 'Other']

### Set path
MODEL_PATH = "/content/drive/My Drive/ld-lsi/res/models/erfnet_road.py"
WEIGHT_PATH = "/content/drive/My Drive/ld-lsi/res/weights/weights_erfnet_road.pth"
IMAGE_PATH = "/content/drive/My Drive/ld-lsi/cb86b1d9-7735472c.mp4"

### DBSCAN hyperparameters
RESIZE_FACTOR = 5
EPS = 1
MIN_SAMPLES = 5
THRESHOLD_POINTS = 700
MAX_WORKERS = 4
MULTITHREADING = True

DEBUG = True

class ClusterExecutor:
    """
        Class used to extract clusters and to handle threading.
    """
    
    def __init__(self, eps, min_samples, threshold_points, multithreading=False, max_workers=5):
        """
                    Class constructor

            Args:
                eps: epsilon used for DBSCAN
                min_samples: number of points required to be a cluster for DBSCAN
                threshold_points: number of points required to be a lane
                multithreading: enable multithreading via ThreadPoolExecutor
                max_workers: number of threads in the pool
        """
        
        # Clustering paramaters
        self.eps = eps
        self.min_samples = min_samples
        self.threshold_points = threshold_points
        
        # Multithreading parameters
        self.multithreading = multithreading
        if(self.multithreading):
            self.threadpool = ThreadPoolExecutor(max_workers=max_workers)
            
    def cluster(self, points):
        """
            Method used to cluster points given by the CNN
            
            Args:
                points: points to cluster. They can be the points classified as egolane, the one classified
                        as other_lane, but NOT together
                        
            Returns:
                pts: an array of arrays containing all the convex hulls (polygons) extracted for that group of points
        """
        
        pts = []
        # Check added to handle when the network doesn't detect any points
        if len(points > 0):
            
            # DBSCAN clustering
            db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
            
            # This is an array of True values to ease class mask calculation
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[np.arange(len(db.labels_))] = True
            
            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
            unique_labels = set(db.labels_)
            
            # Check if we have a cluster for hull extraction
            if n_clusters > 0:
                # Getting a boolean values array representing the pixels belonging to one cluster
                for index, k in enumerate(unique_labels):
                    class_mask = points[core_samples_mask & (db.labels_ == k)]
                    # Filtering clusters too small
                    if class_mask.size > self.threshold_points:
                        # If all the previous checks pass, then we get the hull curve to extract a polygon representing the lane
                        hull = ConvexHull(class_mask)
                        pts.append(np.vstack((class_mask[hull.vertices,1], class_mask[hull.vertices,0])).astype(np.int32).T)
        return pts
    
    def process_cnn_output(self, to_cluster):
        """
            Connector to handle the output of the CNN

            Args: 
                to_cluster: list of arrays of points. In to_cluster[0] there are the points classified by the CNN as egolane,
                            in to_cluster[1] the others. This have to be size 2
            Returns:
                clusters:     list of arrays of points. Here are saved the convex hull extracted. Again, in the first position
                            we find the hulls for the egolane, in the other for the other lanes
        """
        # Output containing the clusters
        clusters = [None] * 2

        ### Multithreading code 
        # We execute in a different thread the clustering for egolane and other lane, 
        # then synchronize the two thread using as_completed. Note that in the pool there are other
        # threads, not synchronized to the two used, ready to start the elaboration for another frame
        if(self.multithreading):
            # Has to use a dict to avoid ambiguity between egolane and otherlanes. If a simple list is used, you can have
            # synchronization errors and put in clusters[0] the other lanes
            futures = {self.threadpool.submit(self.cluster, points) : index for index, points in enumerate(to_cluster)}
            for future in as_completed(futures):
                index = futures[future]
                clusters[index] = future.result()
        ### Single thread
        # If multithreading is disabled, we simply process the points sequentially
        else:
            clusters.append(self.cluster(to_cluster[0]))
            clusters.append(self.cluster(to_cluster[1]))

        return clusters        
        
class LaneExtractor:
    """
        Class used to transform a group of convex hulls of different classes in egolane, right lane and left lane
    """

    def __init__(self):
        """
        Class constructor
        """
        pass

    def get_lanes(self, egolane_clusters, other_lanes_clusters):
        """ 
            Method used to transform the hulls into polygons.
            The first thing to do is to select which of the clusters of the egolane is actually the egolane. 
            This is done selecting the cluster with the biggest area.
            Then, we subtract the intersections with the other lanes to avoid that one pixel is associated to both the egolane
            and another lane.
            Finally, we split the other lanes in left ones and right ones, basing the assumption on the centroid position.
            The biggest cluster on the right will be the right lane, the biggest on the left the left lane.

            args:
                egolane_clusters: set of points that represents the convex hull of the egolane. Can be more than one
                other_lanes_clusters: set of points that represents the convex hull of the other lanes. Can be more than one
        """ 
        ### Slecting the egolane
        egolane_polygons = [Polygon(x) for x in egolane_clusters]
        egolane = max(egolane_polygons, key=lambda p : p.area)
        
        egolane = Polygon(egolane)
        other_lanes_polygons = []
        
        ### Subtracting the intersecting pixels
        # note that this code gives priority to the detection of the other lanes; in this way we can minimize this risk of
        # getting on another lane
        for elem in other_lanes_clusters:
            elem = Polygon(elem)
            other_lanes_polygons.append(elem)
            egolane = egolane.difference(egolane.intersection(elem))
            
        ### Egolane refinement
        # The deletion of the intersecting regions can cause a split in the egolane in more polygons. In this case,
        # the biggest polygon is selected as the new egolane
        if isinstance(egolane, MultiPolygon):
            polygons = list(egolane)
            egolane = max(polygons, key=lambda p : p.area)
            
        ### Spliting the other lanes in left and right ones
        left_lanes = [lane for lane in other_lanes_polygons if lane.centroid.x < egolane.centroid.x]
        right_lanes = [lane for lane in other_lanes_polygons if lane.centroid.x >= egolane.centroid.x]    
        
        ### Selecting the right and the left lane
        left_lane = None if len(left_lanes) == 0 else max(left_lanes, key=lambda p : p.area)
        right_lane = None if len(right_lanes) == 0 else max(right_lanes, key=lambda p : p.area)
        
        ### Numpy conversion
        if egolane is not None: egolane = np.asarray(egolane.exterior.coords.xy).T
        if left_lane is not None: left_lane = np.asarray(left_lane.exterior.coords.xy).T
        if right_lane is not None: right_lane = np.asarray(right_lane.exterior.coords.xy).T

        return egolane, left_lane, right_lane

      
class InferenceLDCNN:
    """
        CNN Node. It takes an image as input and process it using the neural network.
        Then it resizes the output
    """
    def img_received_callback(self, image_path):
        '''
            Callback for image processing
            It submits the image to the CNN, extract the output, then resize it for clustering
        
                Args:
                    image: input image from a video
        '''
        
        cap = cv2.VideoCapture(image_path)
        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (320,240))
        
        while(cap.isOpened()):
            ret, image = cap.read()
            start_t = time.time()
            if ret == True:
                input_tensor = torch.from_numpy(image)
                input_tensor = torch.div(input_tensor.float(), 255)
                input_tensor = input_tensor.permute(2,0,1).unsqueeze(0)
                
                try:
                    ### Pytorch 0.4.0 compatibility inference code
                    if torch.__version__ < "0.4.0":
                        input_tensor = Variable(input_tensor, volatile=True).cuda()
                        output = self.cnn(input_tensor)
                    else:
                        with torch.no_grad():
                            input_tensor = Variable(input_tensor).cuda()
                            output = self.cnn(input_tensor)
                        
                    output, output_road = output
                    road_type = output_road.max(dim=1)[1][0]
                    ### Classification
                    output = output.max(dim=1)[1]
                    output = output.float().unsqueeze(0)    
                    
                    
                    ### Resize to desired scale for easier clustering
                    output = F.interpolate(output, size=(int(output.size(2) / RESIZE_FACTOR), int(output.size(3) / RESIZE_FACTOR)) , mode='nearest')
        
                    ### Obtaining actual output
                    ego_lane_points = torch.nonzero(output.squeeze() == 1)
                    other_lanes_points = torch.nonzero(output.squeeze() == 2)
        
                    ego_lane_points = ego_lane_points.view(-1).cpu().numpy()
                    other_lanes_points = other_lanes_points.view(-1).cpu().numpy()
        
                except Exception as e:
                    print("Can't obtain output. Exception: %s" % e)
                
                ### Post processing
                egolane_points = np.reshape(ego_lane_points, (-1, 2))
                otherlanes_points = np.reshape(other_lanes_points, (-1, 2))
                clusters = self.cexe.process_cnn_output([egolane_points, otherlanes_points])
                
                try:
                    # Classify to 3 lane types
                    ego_lane, left_lane, right_lane = self.le.get_lanes(clusters[0], clusters[1])
                except Exception as e:
                    print("No egolane detected. Sending None for everyone. Exception %s" % e)
                    ego_lane = None
                    left_lane = None
                    right_lane = None
                
                # Logging speed and extracted data
                self.time.append(time.time() - start_t)
                fps = float(len(self.time)) / sum(self.time)
                
                ### Visualization
                try:
                    if DEBUG:
                        lanes_image = np.zeros((70,125,3))
                        
                        if ego_lane is not None:
                            cv2.fillPoly(lanes_image, np.array([ego_lane], dtype=np.int32), COLORS[0])
                        if left_lane is not None:
                            cv2.fillPoly(lanes_image, np.array([left_lane], dtype=np.int32), COLORS[1])
                        if right_lane is not None:
                            cv2.fillPoly(lanes_image, np.array([right_lane], dtype=np.int32), COLORS[2])
                        
                        # Blend the original image and the output of the CNN
                        lanes_image = cv2.resize(lanes_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                        image = cv2.addWeighted(image.astype(float), 1, lanes_image, 0.4, 0)
                        
                        cv2.putText(image, ROAD_MAP[road_type], (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        cv2.putText(image, str(fps), (520,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        image = cv2.resize(image, (320,240), cv2.INTER_NEAREST)
                        image = np.uint8(image)
                        out.write(image)
                        #cv2_imshow("Output", cv2.resize(image, (320,240), cv2.INTER_NEAREST))
                        cv2.waitKey(1)
        
                except Exception as e:
                    print("Visualization error: %s " % e)
            else:
                break        
        cap.release()
        out.release()

    def __init__(self):
        """
            Class constructor
        """
        try:
            self.cexe = ClusterExecutor(EPS, MIN_SAMPLES, THRESHOLD_POINTS, MULTITHREADING, MAX_WORKERS)
            self.le = LaneExtractor()
        except Exception as e:
            print("Something went wrong initializing the ClusterExecutor. Exception: %s" % e)
        
        try:
            spec = importlib.util.spec_from_file_location("erfnet_road",MODEL_PATH)
            erfnet_road = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(erfnet_road)
            self.cnn = erfnet_road.Net()
            #self.cnn = importlib.import_module('ld-lsi.res.models.erfnet_road').Net()
            
            # GPU only mode, setting up
            self.cnn = torch.nn.DataParallel(self.cnn).cuda()
            self.cnn.load_state_dict(torch.load(WEIGHT_PATH))
            self.cnn.eval()
            
        except Exception as e:
            print("Can't load neurol network. Exception: %s" % e)
            
        
        self.time = []
        self.img_received_callback(IMAGE_PATH)
        
        
        
if __name__ == '__main__':
    LDCNN = InferenceLDCNN()
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        