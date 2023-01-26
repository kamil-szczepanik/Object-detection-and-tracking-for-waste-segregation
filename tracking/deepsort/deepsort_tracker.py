import multiprocessing as mp
import queue
import time
import cv2
import torch
import numpy as np
from .deepsort_utils import convert_detections_for_deepsort
from deep_sort_realtime.deepsort_tracker import DeepSort


def DeepSORT_tracker(frameQueue, detections_queue, result_queue,allow_tracker,end_tracker,fps,img_scale_percent,tracker_ready):

    print('tracker:', mp.current_process())

    mot_tracker = DeepSort(max_age=3, max_cosine_distance=0.3, n_init=1, nms_max_overlap=0.35, img_scale_percent=img_scale_percent)
    mot_tracker.tracker.max_iou_distance=0.7 # 0.2 , 0.35, 0.5, 0.7

    sampling_time = 1/fps
    dummy_detections =torch.tensor([[258.7459, 304.2128, 342.2893, 343.4235,   0.8144,   1.0000],
                                    [405.6260, 137.3414, 519.5634, 293.4001,   0.7609,   3.0000],
                                    [546.6186,  79.7919, 657.5923, 165.6807,   0.7551,   1.0000],
                                    [436.6346,  74.1342, 547.4369, 163.3923,   0.7205,   1.0000]])
    dets = convert_detections_for_deepsort(dummy_detections)
    width = int(1920 * img_scale_percent / 100)
    height = int(1080 * img_scale_percent / 100)
    dim = (width, height)

    dummy_frame = np.zeros([height,width,3],dtype=np.uint8)
    dummy_frame = cv2.resize(dummy_frame, dim, interpolation = cv2.INTER_AREA)

    start_dummy = time.time()
    mot_tracker.update_tracks(dets, frame=dummy_frame)
    mot_tracker.tracker.tracks = []

    tracker_ready.set()
    detection = None
    while True:
        
        tracker_start = time.time()

        if not allow_tracker.is_set():
            continue

        try:

            tracking_frame = frameQueue.get_tracking_frame()
        except queue.Empty:
            print('empty')
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
                    dets = convert_detections_for_deepsort(detection[2])
                    tracks = mot_tracker.update_tracks(dets, frame=detection[0])
                    tracked_objects = []
                    for track_obj in tracks:
                        (x1, y1, x2, y2), obj_id, cls_pred = track_obj.to_ltrb(), track_obj.track_id, track_obj.get_det_class()
                        tracked_objects.append(((x1, y1, x2, y2), obj_id, cls_pred))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, detection))
                    detection = None
                else:
                    print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                    tracks = mot_tracker.update_tracks([], frame=tracking_frame[0])
                    tracked_objects = []
                    for track_obj in tracks:
                        (x1, y1, x2, y2), obj_id, cls_pred = track_obj.to_ltrb(), track_obj.track_id, track_obj.get_det_class()
                        tracked_objects.append(((x1, y1, x2, y2), obj_id, cls_pred))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, None))

            else: #  detection!=None
                if detection[1] == tracking_frame[1]:
                    print(f"Tracker: Running tracking WITH detections {detection[1]} on frame {tracking_frame[1]}")
                    dets = convert_detections_for_deepsort(detection[2])
                    tracks = mot_tracker.update_tracks(dets, frame=detection[0])
                    tracked_objects = []
                    for track_obj in tracks:
                        (x1, y1, x2, y2), obj_id, cls_pred = track_obj.to_ltrb(), track_obj.track_id, track_obj.get_det_class()
                        tracked_objects.append(((x1, y1, x2, y2), obj_id, cls_pred))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, detection))
                    detection = None

                else:
                    print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                    tracks = mot_tracker.update_tracks([], frame=tracking_frame[0])
                    tracked_objects = []
                    for track_obj in tracks:
                        (x1, y1, x2, y2), obj_id, cls_pred = track_obj.to_ltrb(), track_obj.track_id, track_obj.get_det_class()
                        tracked_objects.append(((x1, y1, x2, y2), obj_id, cls_pred))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, None))
        else: # detections_queue.qsize() == 0
            if detection != None:
                if detection[1] == tracking_frame[1]:
                    print(f"Tracker: Running tracking WITH detections {detection[1]} on frame {tracking_frame[1]}")
                    dets = convert_detections_for_deepsort(detection[2])
                    tracks = mot_tracker.update_tracks(dets, frame=detection[0])
                    tracked_objects = []
                    for track_obj in tracks:
                        (x1, y1, x2, y2), obj_id, cls_pred = track_obj.to_ltrb(), track_obj.track_id, track_obj.get_det_class()
                        tracked_objects.append(((x1, y1, x2, y2), obj_id, cls_pred))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, detection))
                    detection = None
                else:
                    print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                    tracks = mot_tracker.update_tracks([], frame=tracking_frame[0])
                    tracked_objects = []
                    for track_obj in tracks:
                        (x1, y1, x2, y2), obj_id, cls_pred = track_obj.to_ltrb(), track_obj.track_id, track_obj.get_det_class()
                        tracked_objects.append(((x1, y1, x2, y2), obj_id, cls_pred))
                    print(f"Tracker: Done frame {tracking_frame[1]}")
                    result_queue.put((tracking_frame, tracked_objects, None))
            else:
                print(f"Tracker: Running tracking WITHOUT detections on frame {tracking_frame[1]}")
                tracks = mot_tracker.update_tracks([], frame=tracking_frame[0])
                tracked_objects = []
                for track_obj in tracks:
                    (x1, y1, x2, y2), obj_id, cls_pred = track_obj.to_ltrb(), track_obj.track_id, track_obj.get_det_class()
                    tracked_objects.append(((x1, y1, x2, y2), obj_id, cls_pred))
                print(f"Tracker: Done frame {tracking_frame[1]}")
                result_queue.put((tracking_frame, tracked_objects, None))

        diff = time.time() - tracker_start
        while  diff < sampling_time and frameQueue.qsize()<=3:
            diff = time.time() - tracker_start
        print('tracker time:', diff)


    # cleaning detections queue
    end_tracker.set()
    while detections_queue.qsize()>0:
        detections_queue.get()

    while result_queue.qsize()>0:
        time.sleep(1)
        continue

    print(' === Tracker off === ')