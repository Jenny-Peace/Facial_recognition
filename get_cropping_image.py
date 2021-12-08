import cv2, numpy as np, pandas as pd, os, sys 
import matplotlib.pyplot as plt
from datetime import datetime


# SETUP CAMERA >> ADD MEDIA FOLDER PATH TO SHELL
cap = cv2.VideoCapture(0)

if not cap.isOpened():  # Check if the webcam is opened correctly
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()

    # LOAD MODEL YOLOv3
    MODEL = 'YOLO model/yolov3-face.cfg'
    WEIGHT = 'YOLO model/yolov3-wider_16000.weights'
    
    net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT) 
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


    ### DETECT IMAGE BY USING BLOB
    IMG_WIDTH, IMG_HEIGHT = 416, 416   #input image dimensions
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(IMG_WIDTH, IMG_HEIGHT), mean=[0, 0, 0], swapRB=1, crop=False)
    net.setInput(blob)                                      
    output_layers = net.getUnconnectedOutLayersNames()      
    outs = net.forward(output_layers)                     


    # DETECT HIGHEST CONFIDENCE SCORES
    frame_height    = frame.shape[0]
    frame_width     = frame.shape[1]
            
    confidences = []
    boxes = []
            
    for out in outs:                                         
        for detection in out:                                
            confidence = detection[-1]
            if confidence > 0.5:                            
                center_x    = int(detection[0] * frame_width)
                center_y    = int(detection[1] * frame_height)
                width       = int(detection[2] * frame_width)  
                height      = int(detection[3] * frame_height)
                
                topleft_x   = int(center_x - width/2)  # Find the top left point of the bounding box 
                topleft_y   = int(center_y - height/2)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)    # Perform non-maximum suppression to eliminate  # redundant overlapping boxes with lower confidences.

    # DRAWING BOUNDING BOX OF OBJECT DETECTION
    final_boxes = []
    result      = frame.copy()

    for i in indices:
        i   = i[0]
        box = boxes[i]
        final_boxes.append(box)
                
        left    = box[0]            
        top     = box[1]
        width   = box[2]
        height  = box[3]

        cv2.rectangle(result, (left, top), (left + width + 5, top + height + 5), (0,0,255), 2)         
        
        text = f'{confidences[i]:.2f}'                                                          
        cv2.putText(result, text, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 190, 200), 2)  
    
        face_quantity = f'Detected face(s): {len(final_boxes)}'                                 
        cv2.putText(result, face_quantity, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow('face detection:', result)

    
    c = cv2.waitKey(1)
    if c == 27:                 # ESCAPE button for breaking capturing
        break

    elif  c == ord("s"):        # 'S' button for breaking capturing
          try: 
              for box in final_boxes:
                crop = frame[(top - 35):(top + height + 35),(left - 35):(left + width + 35)] 
                image = cv2.resize(crop, (300, 300), interpolation = cv2.INTER_AREA)
                image = image / 255.
                image = image.flatten()
                now = datetime.now().strftime("%H%M%S") # for window, use: datetime.datetime.now().strftime("%H%M%S")
                cv2.imwrite(f'DataNew1/Jenny{now}.jpg',crop)  #path of saved image
                
          except Exception as ex:
              print(ex)

cap.release()
cv2.destroyAllWindows()
