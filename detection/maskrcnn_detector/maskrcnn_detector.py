from .detector_utils import detect_image_maskrcnn
import queue
import multiprocessing as mp
import numpy as np
import time



def MaskRCNN_detector(frameQueue, detections_queue, allow_tracker,end_detector, predictor,detector_ready,end_tracker,img_scale_percent):
    print('detector:', mp.current_process())

    width = int(1920 * img_scale_percent / 100)
    height = int(1080 * img_scale_percent / 100)
    dummy_frame = np.zeros([height,width,3],dtype=np.uint8)
    start_dummy = time.time()
    detections_output, model_output = detect_image_maskrcnn(dummy_frame, predictor)
    detector_ready.set()
    
    while True:
        
        if frameQueue.qsize() > 0:
            break

    while True:
        detector_start = time.process_time()
        try:
            frame, frame_num = frameQueue.get_detection_frame()
        except queue.Empty:
            print('Detector: no more frames to detect')
            break

        print(f"Detector: Running detection on frame {frame_num}")
        ###################
        det_start = time.process_time()
        detections_output, model_output = detect_image_maskrcnn(frame, predictor)
        # DETECTION PROCESS
        ###################
        print(f"Detector: Done frame {frame_num}")
        if allow_tracker.is_set()==False:
            allow_tracker.set()
        detections_queue.put((frame, frame_num, detections_output, model_output))
        detector_stop = time.process_time()
        print('Detector: loop time =', detector_stop - detector_start)
    end_detector.set()

    while not end_tracker.is_set():
        time.sleep(1)

    print(' === Detector off === ')