import cv2 as cv
import numpy as np
import os

WORKING_DIR = "/home/ehsan/Workspace/DebitCard/"
CAR_PLATE_DETECTION_CFG_PATH = "yolov3-tiny.cfg"
id_weight_path = "yolov3-tiny_last.weights"

np.random.seed(42)
id_network = cv.dnn.readNetFromDarknet(os.path.join(WORKING_DIR, CAR_PLATE_DETECTION_CFG_PATH),
                                       os.path.join(WORKING_DIR, id_weight_path))

car_plate_layer_names = id_network.getLayerNames()
car_plate_layer_names = [car_plate_layer_names[i[0] - 1] for i in id_network.getUnconnectedOutLayers()]

input_dir = os.listdir("Card/")
for item in input_dir:
    card_image = cv.imread("Card/" + item)
    print(">>> ", card_image.shape)
    (H, W) = card_image.shape[:2]

    try:
        blob = cv.dnn.blobFromImage(card_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        id_network.setInput(blob)
    except:
        continue

    outputs = id_network.forward(car_plate_layer_names)

    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.6:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)

    ids = []
    if len(idxs) > 0:
        # Classify cars and plate in separate lists
        for i in idxs.flatten():
            if classIDs[i] == 0:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                ids.append([x, y, w, h, -1])

    if len(ids) > 0:
        print(ids)
        ids = ids[0]
        card_idd = card_image[ids[1]:ids[1]+ids[3], ids[0]-10:ids[0]+ids[2]+10]
        cv.imwrite("Out/" + item, card_idd)
        # card_image = cv.rectangle(card_image, (ids[0], ids[1]), (ids[0] + ids[2], ids[1] + ids[3]), (255, 0, 0), 2)
        # cv.imshow("", card_image)
        # cv.waitKey()
    else:
        print("Not found!!!")