import multiprocessing as mp
import cv2
import time
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode

def KCF_system(path_to_video, frameQueue, detections_queue, result_queue, allow_tracker, end_detector, end_tracker,dataset_metadata_catalog,detector_ready,fps,img_scale_percent,tracker_ready):

    print('system:', mp.current_process())
    ########### Video ###################
    video = cv2.VideoCapture(path_to_video)
    if (video.isOpened() == False): 
        print("Error reading video file")
    sampling_time = 1/fps
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    width = int(frame_width * img_scale_percent / 100)
    height = int(frame_height * img_scale_percent / 100)
    dim = (width, height)
    ###################################
    result = cv2.VideoWriter(f"kcf_tracker-scale{img_scale_percent}-{fps}fps.avi", 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            fps, dim)
    ###################################
    class_path='config/zerowaste.names'
    counter = 1
    system_start = time.time()
    while(True):
        if not detector_ready.is_set():
            continue

        start = time.time()

        ret, frame = video.read()
        if ret == True:
            print('------------------ ' + str(counter) +' ------------------')
            resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            frameQueue.add_frame((resized_frame, counter))
            frameQueue.print()

            if allow_tracker.is_set() == False:      
                print('--> Add to queue and go to next frame')
                counter += 1
                diff = time.time() - start

                while  diff < sampling_time:
                    diff = time.time() - start
                print(diff) 
                continue

        elif ret == False:
            break
        
        counter += 1

        diff = time.time() - start
        while  diff < sampling_time:
            diff = time.time() - start
        print(diff)
        
    while True:
        if end_detector.is_set() and end_tracker.is_set():
            break
    system_stop = time.time()
    print('System: loop time =', system_stop-system_start)
    

    v = VideoVisualizer(dataset_metadata_catalog, ColorMode.IMAGE)
    print('System: Saving frames to video')
    while result_queue.qsize()>0: 
        (tracking_frame, tracking_frame_num), tracked_objects, detection = result_queue.get()
        if detection is not None:
            detection_frame, detection_frame_num, detections_output, model_output = detection
            visualizer = v.draw_instance_predictions(detection_frame, model_output["instances"])
            img_pred = visualizer.get_image()
            cv2.putText(img_pred,  "DETECTION", (5, 40), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0,255,0), 1)
            tracking_frame = img_pred
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(tracking_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(tracking_frame, cls_pred + "-" + str(int(obj_id)), (x1, y1 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255), 1)

        cv2.putText(tracking_frame,  f"FRAME: {tracking_frame_num} | FPS: {fps}", (5, 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
        result.write(tracking_frame)


    result.release()
    video.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    print("The video was successfully saved")