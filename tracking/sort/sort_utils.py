def x1y1wh_to_x1y1x2y2(x1,y1,w,h):
    x2, y2 = x1+w, y1+h
    return x1, y1, x2, y2

def x1y1x2y2_to_x1y1wh(x1, y1, x2, y2):
    w =  x2- x1
    h = y2 - y1
    return x1, y1, w, h

def convert_detections_for_deepsort(detections):
    result = []
    for detection in detections:
        x1, y1, x2, y2 = detection[:4]
        x1, x2, w, h = x1y1x2y2_to_x1y1wh(x1.detach().numpy(), y1.detach().numpy(), x2.detach().numpy(), y2.detach().numpy())
        result.append(([x1, x2, w, h], detection[4].detach().numpy(), detection[5].detach().numpy()))
    return result


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

