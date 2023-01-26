from .KCFMultiTracker import MultiTracker
import multiprocessing as mp
import queue
import time
import cv2

def KCF_tracker(frameQueue,detections_queue,result_queue,allow_tracker,end_tracker,fps,img_scale_percent,tracker_ready):

    mot_tracker = MultiTracker(max_age=2, iou_threshold=0.3)

    sampling_time = 1/fps
    print('tracker KCF:', mp.current_process())
    detect_thresh = 0.4
    param_handler = cv2.TrackerKCF_Params()
    setattr(param_handler, "detect_thresh", detect_thresh)
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
                    tracking_boxes, obj_ids, pred_class_names = mot_tracker.update(tracking_frame[0], detection[2])
                    
                    tracked_objects = []
                    for box, obj_id, pred_class_name in zip(tracking_boxes, obj_ids, pred_class_names):
                        (x, y, w, h) = [int(v) for v in box]
                        tracked_objects.append((x, y, x+w, y+h, obj_id, pred_class_name))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, detection))
                    detection = None
                else:
                    print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                    tracking_boxes, obj_ids, pred_class_names = mot_tracker.update(tracking_frame[0])
                    tracked_objects = []
                    for box, obj_id, pred_class_name in zip(tracking_boxes, obj_ids, pred_class_names):
                        (x, y, w, h) = [int(v) for v in box]
                        tracked_objects.append((x, y, x+w, y+h, obj_id, pred_class_name))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, None))

            else: #  detection!=None
                if detection[1] == tracking_frame[1]:
                    print(f"Tracker: Running tracking WITH detections {detection[1]} on frame {tracking_frame[1]}")
                    tracking_boxes, obj_ids, pred_class_names = mot_tracker.update(tracking_frame[0], detection[2] )
                    tracked_objects = []
                    for box, obj_id, pred_class_name in zip(tracking_boxes, obj_ids, pred_class_names):
                        (x, y, w, h) = [int(v) for v in box]
                        tracked_objects.append((x, y, x+w, y+h, obj_id, pred_class_name))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, detection))
                    detection = None

                else:
                    print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                    tracking_boxes, obj_ids, pred_class_names = mot_tracker.update(tracking_frame[0])
                    tracked_objects = []
                    for box, obj_id, pred_class_name in zip(tracking_boxes, obj_ids, pred_class_names):
                        (x, y, w, h) = [int(v) for v in box]
                        tracked_objects.append((x, y, x+w, y+h, obj_id, pred_class_name))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, None))
        else: # detections_queue.qsize() == 0
            if detection != None:
                if detection[1] == tracking_frame[1]:
                    print(f"Tracker: Running tracking WITH detections {detection[1]} on frame {tracking_frame[1]}")
                    tracking_boxes, obj_ids, pred_class_names = mot_tracker.update(tracking_frame[0], detection[2])
                    tracked_objects = []
                    for box, obj_id, pred_class_name in zip(tracking_boxes, obj_ids, pred_class_names):
                        (x, y, w, h) = [int(v) for v in box]
                        tracked_objects.append((x, y, x+w, y+h, obj_id, pred_class_name))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, detection))
                    # result_queue.task_done()
                    detection = None
                else:
                    print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                    tracking_boxes, obj_ids, pred_class_names = mot_tracker.update(tracking_frame[0])
                    tracked_objects = []
                    for box, obj_id, pred_class_name in zip(tracking_boxes, obj_ids, pred_class_names):
                        (x, y, w, h) = [int(v) for v in box]
                        tracked_objects.append((x, y, x+w, y+h, obj_id, pred_class_name))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, None))
            else:
                print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                tracking_boxes, obj_ids, pred_class_names = mot_tracker.update(tracking_frame[0])
                tracked_objects = []
                for box, obj_id, pred_class_name in zip(tracking_boxes, obj_ids, pred_class_names):
                    (x, y, w, h) = [int(v) for v in box]
                    tracked_objects.append((x, y, x+w, y+h, obj_id, pred_class_name))
                print(f"Tracker: Done frame {tracking_frame[1]}")
                result_queue.put((tracking_frame, tracked_objects, None))

        diff = time.time() - tracker_start
        while  diff < sampling_time and frameQueue.qsize()<2:
            diff = time.time() - tracker_start
        print('Tracker: loop time =', diff)


    # cleaning detections
    end_tracker.set()
    while detections_queue.qsize()>0:
        detections_queue.get()

    while result_queue.qsize()>0:
        time.sleep(1)
        continue

    print(' === Tracker off === ')
