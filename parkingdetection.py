from re import L
from turtle import width
import git
import pathlib
import wget
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
#we need to install every specified module including - pip install youtube-dl==2020.12.2
import pafy

def setup_yolo():
    CURRENT_PATH = str(pathlib.Path(__file__).parent.resolve())
    if not os.path.exists(CURRENT_PATH + '/darknet'):
        git.Git(CURRENT_PATH + '/darknet').clone("https://github.com/pjreddie/darknet")

    if not os.path.exists(CURRENT_PATH + '/yolov3.weights'):
        yolo_weights = wget.download('https://pjreddie.com/media/files/yolov3.weights')
    else:
        yolo_weights = 'yolov3.weights'
    
    yolo = cv2.dnn.readNet(yolo_weights,CURRENT_PATH+ '/darknet/cfg/yolov3.cfg')
    classes = []
    with open(CURRENT_PATH+ '/darknet/data/coco.names', 'r') as f:
        classes = f.read().splitlines(True)
    for label in classes: label = label.strip('\n')
    return yolo, classes


def get_output_layers(net):
    
    layer_names = net.getLayerNames()  
    for i in net.getUnconnectedOutLayers():
        output_layers = layer_names[i - 1]

    return output_layers

def main():
    net, classes = setup_yolo()
    url   = "https://www.youtube.com/watch?v=U7HRKjlXK-Y"
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    vid = cv2.VideoCapture(best.url)
    while(True):
      
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0, 0, 0), swapRB=True, crop = False)
        net.setInput(blob)

        outputs = net.forward(get_output_layers(net))

        #boxes
        width = frame.shape[1]
        height = frame.shape[0]
        boxes = []
        confidences = []
        class_ids = []
        for detection in outputs :
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.3 :
                    center_x = int(detection[0]*width)
                    center_h = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_h - h/2)

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
        font = cv2.FONT_HERSHEY_DUPLEX
        colors = np.random.uniform(0, 255, size = (len(boxes),3))

        for i in indexes.flatten():
           x, y, w, h = boxes [i]
           label = str(classes[class_ids[i]])
           confi = str(round(confidences[i], 2))
           color = colors[i]

           cv2.rectangle(frame, (x,y), (x+w, y+h), color, 3)
           cv2.putText(frame, label +""+ confi, (x,y+20), font, 1, (255,255,255), 1)
        
        # Display the resulting frame

        num_of_cars = class_ids.count('car\n')
        num_of_trucks = class_ids.count('truck\n')
        counter = num_of_cars + num_of_trucks

        cv2.imshow('object detection', frame)
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main())
