from scripts.FrameQueue import FrameQueue, MyManager
import multiprocessing as mp
from tracking.kcf import kcf_system, kcf_tracker
from tracking.sort import sort_system, sort_tracker
from tracking.deepsort import deepsort_system, deepsort_tracker
from detection.maskrcnn_detector import maskrcnn_detector

import argparse

from detectron2.utils.logger import setup_logger
setup_logger()
# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances


def detect_and_track_zerowaste(dataset_metadata_catalog, fps=12, img_scale_percent=100, tracking="deepsort", detector_score_threshold=0.65,path_to_video='videos/08_frame-10-12.mp4'):

    MyManager.register('FrameQueue', FrameQueue)

    cfg = get_cfg()
    cfg.merge_from_file(
        "config/detectron2_custom_congif_zerowaste_maskrcnn_R_101_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = detector_score_threshold   # set the testing threshold for this model
    predictor = DefaultPredictor(cfg)

    if tracking == "deepsort":
        tracker = deepsort_tracker.DeepSORT_tracker
        system = deepsort_system.DeepSORT_system
    elif tracking== "sort":
        tracker = sort_tracker.SORT_tracker
        system = sort_system.SORT_system
    elif tracking== "kcf":
        tracker = kcf_tracker.KCF_tracker
        system = kcf_system.KCF_system
    else:
        raise Exception("Incorrect tracking algorithm name")

    detector = maskrcnn_detector.MaskRCNN_detector

    with MyManager() as manager:
        frameQueue = manager.FrameQueue()
        detections_queue = mp.Queue()
        result_queue = mp.Queue()

        allow_tracker = mp.Event()
        end_detector = mp.Event()
        end_tracker = mp.Event()
        detector_ready = mp.Event()
        tracker_ready = mp.Event()

        detector_p = mp.Process(target=detector, args=(frameQueue,detections_queue,allow_tracker,end_detector,predictor,detector_ready,end_tracker,img_scale_percent,))
        detector_p.start()

        tracker_p = mp.Process(target=tracker, args=(frameQueue,detections_queue,result_queue,allow_tracker,end_tracker,fps,img_scale_percent,tracker_ready,))
        tracker_p.start()

        system_p = mp.Process(target=system, args=(path_to_video, frameQueue, detections_queue, result_queue, allow_tracker, end_detector, end_tracker,dataset_metadata_catalog,detector_ready,fps,img_scale_percent,tracker_ready,))
        system_p.start()


        detector_p.join()
        tracker_p.join()
        system_p.join()



if __name__=="__main__":

    parser = argparse.ArgumentParser("Run multiple object detection and tracking")

    parser.add_argument("--tracking", type=str, help="Choose tracking algorithm between 'sort', 'deepsort' and 'kcf'", default="sort")
    parser.add_argument("--fps", type=int, help="FPS of video stream", default=12)
    parser.add_argument("--img_scale_percent", type=int, help="Scale of frames in percent", default=100)
    parser.add_argument("--detector_score_threshold", type=float, help="Score threshold of the detector", default=0.65)
    parser.add_argument("--path_to_dataset", type=str, help="Path to ZeroWaste dataset test folder", default="/content/drive/MyDrive/inzynierka/datasets/zerowaste-f-final/splits_final_deblurred/test/")
    parser.add_argument("--path_to_video", type=str, help="Path to video file", default='videos/08_frame-10-12.mp4')
    args = parser.parse_args()

    path_to_dataset = args.path_to_dataset

    mp.set_start_method("spawn")
    register_coco_instances("zerowaste_test", {}, path_to_dataset+"labels.json", path_to_dataset+"data")
    zerowaste_test_metadata = MetadataCatalog.get("zerowaste_test")
    dataset_dicts_test = DatasetCatalog.get("zerowaste_test")

    detect_and_track_zerowaste(dataset_metadata_catalog=zerowaste_test_metadata, 
                                tracking=args.tracking, 
                                fps=args.fps, 
                                img_scale_percent=args.img_scale_percent,
                                detector_score_threshold=args.detector_score_threshold,
                                path_to_video=args.path_to_video)