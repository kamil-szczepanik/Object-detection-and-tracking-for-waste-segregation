from __future__ import division
import torch


def detect_image_maskrcnn(img, predictor):
    """
    retruns boxes (x1,y1,x2,y2, score, prediction_class)
    """
    with torch.no_grad():
        model_output= predictor(img)
        model_output['instances'] = model_output['instances'].to("cpu")

        boxes = model_output['instances'].get_fields()['pred_boxes'].tensor
        scores = model_output['instances'].get_fields()['scores']
        pred_classes = model_output['instances'].get_fields()['pred_classes']
        pred_masks = model_output['instances'].get_fields()['pred_masks']

        output = []
        for box, score, pred_class in zip(boxes, scores, pred_classes):
            output.append(torch.cat((box,torch.unsqueeze(score, 0),torch.unsqueeze(pred_class, 0)), 0))
        if not len(output) == 0:
            detections = torch.stack(output, 0)
        else:
            detections = []
    return detections, model_output
