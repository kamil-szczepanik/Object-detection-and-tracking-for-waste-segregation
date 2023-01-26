import collections
import numpy as np
import time
import cv2
from .kcf_utils import get_iou, x1y1x2y2_to_x1y1wh, x1y1wh_to_x1y1x2y2


class TrackerStruct():
    def __init__(self, tracker, id, pred_class_name) -> None:
        self.tracker = tracker
        self.id = id
        self.age = 0
        self.det_age = 0
        self.pred_class_name = pred_class_name
        self.changed_position = None
        self.box_x1y1wh = None 

class MultiTracker():
    def __init__(self, max_age=3, iou_threshold=0.5, frame_size=(1920, 1080)) -> None:
        self.trackers = []
        self.counter = 0
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.frame_counter = 0
        self.velocity = None
        self.shifts = collections.defaultdict(lambda: [])
        self.frame_width = frame_size[0]
        self.frame_height = frame_size[1]
        self.detect_thresh = 0.3
        self.param_handler = cv2.TrackerKCF_Params()
        setattr(self.param_handler, "detect_thresh", self.detect_thresh)


    def update(self, frame, dets = None):

            trackers_to_delete_list = []
            if len(self.trackers) > 0:
                update_tracks_start = time.time()
                for idx in range(len(self.trackers)):
                    tracker_struct = self.trackers[idx]
                    success, box_x1y1wh = tracker_struct.tracker.update(frame)
                    if success is False:
                        tracker_struct.age += 1
                    else:
                        tracker_struct.age = 0
                    if tracker_struct.age > self.max_age: # deleting old trackers
                        trackers_to_delete_list.append(tracker_struct)
                        continue
                    if tracker_struct.det_age > self.max_age: # deleting old trackers
                        trackers_to_delete_list.append(tracker_struct)
                        continue
                    if box_x1y1wh != (0,0,0,0):
                        tracker_struct.age = 0
                        tracker_struct.det_age = 0
                    else:
                        trackers_to_delete_list.append(tracker_struct)
                        continue
                    
                        
                    tracker_struct.box_x1y1wh = box_x1y1wh
                    tracker_struct.det_age+=1

                for t in trackers_to_delete_list:
                    self.trackers.remove(t)

                # adding new detections
                add_dets_start = time.time()
                if dets is not None:
                    #print('det len:', len(dets))
                    
                    for det in dets:
                        replaced = False
                        new_box_x1y1x2y2 = det[:4]
                        #print('DET:',new_box_x1y1x2y2)
                        new_pred_class_name = ['rigid_plastic', 'cardboard', 'metal', 'soft_plastic'][int(det[-1])]
                        
                        for tracker_struct_idx in range(len(self.trackers)):
                            tracker_struct = self.trackers[tracker_struct_idx]

                            if tracker_struct.box_x1y1wh != (0,0,0,0):
                                # check if similar tracker already exists
                                box_x1y1x2y2 = x1y1wh_to_x1y1x2y2(tracker_struct.box_x1y1wh[0], tracker_struct.box_x1y1wh[1], tracker_struct.box_x1y1wh[2], tracker_struct.box_x1y1wh[3])
                                iou = get_iou(new_box_x1y1x2y2, box_x1y1x2y2)

                                if iou > self.iou_threshold and new_pred_class_name==tracker_struct.pred_class_name:
                                    (x1, y1, x2, y2) = [int(v) for v in new_box_x1y1x2y2]
                                    new_box_x1y1wh = x1y1x2y2_to_x1y1wh(x1, y1, x2, y2)
                                    tracker = cv2.TrackerKCF_create(self.param_handler)
                                    tracker.init(frame, new_box_x1y1wh)
                                    new_tracker_struct = TrackerStruct(tracker, tracker_struct.id, tracker_struct.pred_class_name)
                                    new_tracker_struct.box_x1y1wh = new_box_x1y1wh
                                    self.trackers[tracker_struct_idx] = new_tracker_struct
                                    #box_list[tracker_struct_idx] = new_box_x1y1wh
                                    #print('replaced', new_tracker_struct.id)
                                    
                                    replaced = True
                                    break
                              
                        if replaced == False: # not rapleced so we add new object
                            #print('no similar existing track - adding new')
                            (x1, y1, x2, y2) = [int(v) for v in new_box_x1y1x2y2]
                            new_box_x1y1wh = x1y1x2y2_to_x1y1wh(x1, y1, x2, y2)
                            tracker = cv2.TrackerKCF_create(self.param_handler)
                            tracker.init(frame, new_box_x1y1wh)
                            new_tracker_struct = TrackerStruct(tracker, self.counter, new_pred_class_name)
                            new_tracker_struct.box_x1y1wh = new_box_x1y1wh
                            self.trackers.append(new_tracker_struct)
                            #print('added new', new_tracker_struct.id)


                            self.counter += 1
                #print('adding dets time=', time.time()-add_dets_start )

                           

            else: # if self.trackers is empty so there's nothing to update and we add new tracks
                if dets is not None:
                    for det in dets:
                        new_box = det[:4]
                        new_pred_class_name = ['rigid_plastic', 'cardboard', 'metal', 'soft_plastic'][int(det[-1])]
                        (x1, y1, x2, y2) = [int(v) for v in new_box]
                        new_box_x1y1wh = x1y1x2y2_to_x1y1wh(x1, y1, x2, y2)
                        tracker = cv2.TrackerKCF_create(self.param_handler)
                        tracker.init(frame, new_box_x1y1wh)
                        new_tracker_struct = TrackerStruct(tracker, self.counter, new_pred_class_name)
                        new_tracker_struct.box_x1y1wh = new_box_x1y1wh
                        self.trackers.append(new_tracker_struct)
                        self.counter += 1
                        
                        #print('added to empty', new_tracker_struct.id)

            new_box_list, new_id_list, new_pred_class_names_list = [], [], []
            for t in self.trackers:
                #b = t.box_x1y1wh
                s, b = t.tracker.update(frame)
                new_box_list.append(b)
                new_id_list.append(t.id)
                new_pred_class_names_list.append(t.pred_class_name)

            return new_box_list, new_id_list, new_pred_class_names_list
        
