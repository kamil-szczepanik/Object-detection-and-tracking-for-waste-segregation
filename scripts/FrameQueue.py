import queue
from multiprocessing.managers import BaseManager

class FrameQueue(queue.Queue):
    def __init__(self):
        super().__init__()
        self.detector_LIFO_queue = queue.LifoQueue()

    def add_frame(self, frame):
        self.put(frame)
        while not self.detector_LIFO_queue.empty():
            try:
                self.detector_LIFO_queue.get(False)
            except queue.Empty:
                continue
            self.detector_LIFO_queue.task_done()

        self.detector_LIFO_queue.put(frame)

    def get_detection_frame(self):
        return self.detector_LIFO_queue.get(timeout=2)

    def get_tracking_frame(self): return self.get(timeout=2)

    def print(self):
        print([element[1] for element in self.queue])
        # return print()
    
    def print_det_size(self):
        print(self.detector_LIFO_queue.qsize())


class MyManager(BaseManager):
    pass