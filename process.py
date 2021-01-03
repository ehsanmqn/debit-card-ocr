import os
import numpy as np
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model


yolo_config_path = os.path.abspath("yolov3-tiny.cfg")
digit_weights_path = os.path.abspath("digit.weights")
id_weight_path = os.path.abspath("id.weights")
digit_network = cv2.dnn.readNetFromDarknet(yolo_config_path, digit_weights_path)
id_network = cv2.dnn.readNetFromDarknet(yolo_config_path, id_weight_path)


def detect_id(card_image, threshold):
    (H, W) = card_image.shape[:2]
    car_plate_layer_names = id_network.getLayerNames()
    car_plate_layer_names = [car_plate_layer_names[i[0] - 1] for i in id_network.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(card_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    id_network.setInput(blob)
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

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.3)

    ids = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            if classIDs[i] == 0:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                ids.append([x, y, w, h, -1])

    if len(ids) > 0:
        ids = ids[0]
        return ids
    else:
        return []


def segment_id(id_image, threshold):

    (H, W) = id_image.shape[:2]
    ln = digit_network.getLayerNames()
    ln = [ln[i[0] - 1] for i in digit_network.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(id_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    digit_network.setInput(blob)

    layerOutputs = digit_network.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.3)

    aa = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            a = [
                int(classIDs[i]),
                confidences[i],
                boxes[i][0],
                boxes[i][1],
                boxes[i][2],
                boxes[i][3],
            ]

            aa.append(a)

    if len(aa) > 0:
        aa = np.array(aa)
        sorted_array = aa[np.argsort(aa[:, 2])]
        croped_image = []

        for i in range(len(sorted_array)):
            croped_image.append(
                id_image[
                int(sorted_array[i, 3])
                - 1: int(sorted_array[i, 3] + sorted_array[i, 5] + 1),
                int(sorted_array[i, 2])
                - 1: int(sorted_array[i, 2] + sorted_array[i, 4] + 1),
                ]
            )

        return croped_image, sorted_array

    return []

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="mnist.tflite")
interpreter.allocate_tensors()

digit_model = load_model('cnn.h5')

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_dir = os.listdir("Cards/")

counter = 0
for filename in input_dir:
    card_image = cv2.imread("Cards/" + filename)

    id_shape = detect_id(card_image, 0.3)



    if len(id_shape) > 0:
        id_image = card_image[id_shape[1]:id_shape[1]+id_shape[3], id_shape[0]-10:id_shape[0]+id_shape[2]+10]

        try:
            cropped, arr = segment_id(id_image, 0.3)

            print(">> Number of segments: ", len(cropped))

            if len(cropped) > 0:
                for item in cropped:
                    # Test model on random input data.
                    try:
                        digit = np.invert(item)
                        digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
                        # digit = digit.reshape(1, 28, 28, 1)
                        # digit = digit.astype('float32')
                        # digit /= 255
                        # out = digit_model.predict(digit)
                        # print(np.argmax(out))
                        counter += 1
                        cv2.imwrite("Digits/" + str(counter) + ".jpg", digit)

                        input_shape = input_details[0]['shape']
                        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
                        input_data[0] = digit
                        interpreter.set_tensor(input_details[0]['index'], input_data)

                        interpreter.invoke()

                        # The function `get_tensor()` returns a copy of the tensor data.
                        # Use `tensor()` in order to get a pointer to the tensor.
                        output_data = interpreter.get_tensor(output_details[0]['index'])
                        # print(output_data[0], np.argmax(output_data[0]))

                        # cv2.imshow("", item)
                        # cv2.waitKey()
                    except:
                        continue
        except:
            continue