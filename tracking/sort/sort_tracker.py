import multiprocessing as mp
import queue
import time
import numpy as np
from .sort import *



def SORT_tracker(frameQueue,detections_queue,result_queue,allow_tracker,end_tracker,fps,img_scale_percent,tracker_ready):
    mot_tracker = Sort(max_age=6, min_hits=0, iou_threshold=0.2, img_scale_percent=img_scale_percent) 
    sampling_time = 1/fps
    print('tracker:', mp.current_process())
    detection = None
    while True:
        tracker_start = time.time()

        if not allow_tracker.is_set():
            continue
        
        try:
            tracking_frame = frameQueue.get_tracking_frame()
        except queue.Empty:
            break

        if detection != None:
            while detection[1] < tracking_frame[1] and detections_queue.qsize()>0:
                detection = detections_queue.get()

        detections_queue.qsize()
        if detections_queue.qsize()>0:
            if detection==None:
                detection = detections_queue.get()
                if detection[1] == tracking_frame[1]:
                    print(f"Tracker: Running tracking WITH detections {detection[1]} on frame {tracking_frame[1]}")
                    tracked_objects = mot_tracker.update(detection[2])
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, detection))
                    detection = None
                else:
                    print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                    tracked_objects = mot_tracker.update(dets=np.empty((0, 5)))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, None))

            else: #  detection!=None
                if detection[1] == tracking_frame[1]:
                    print(f"Tracker: Running tracking WITH detections {detection[1]} on frame {tracking_frame[1]}")
                    tracked_objects = mot_tracker.update(detection[2])
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, detection))
                    detection = None

                else:
                    print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                    tracked_objects = mot_tracker.update(dets=np.empty((0, 5)))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, None))
        else: # detections_queue.qsize() == 0
            if detection != None:
                if detection[1] == tracking_frame[1]:
                    print(f"Tracker: Running tracking WITH detections {detection[1]} on frame {tracking_frame[1]}")
                    tracked_objects = mot_tracker.update(detection[2])
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, detection))
                    detection = None
                else:
                    print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                    tracked_objects = mot_tracker.update(dets=np.empty((0, 5)))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, None))
            else:
                print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                tracked_objects = mot_tracker.update(dets=np.empty((0, 5)))
                print(f"Tracker: Done frame {tracking_frame[1]}")
                result_queue.put((tracking_frame, tracked_objects, None))

        diff = time.time() - tracker_start
        while  diff < sampling_time and frameQueue.qsize()<=3:
            diff = time.time() - tracker_start
        print('Tracker: loop time:', diff)


    # cleaning detections queue
    end_tracker.set()
    while detections_queue.qsize()>0:
        detections_queue.get()

    while result_queue.qsize()>0:
        time.sleep(1)
        continue

    print(' === Tracker off === ')
