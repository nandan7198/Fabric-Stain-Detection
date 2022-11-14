import numpy as np
import argparse
import time
import cv2
from pathlib import Path
import os
import mysql.connector
import tkinter.filedialog
from tkinter import *
import decimal
decimal.getcontext().rounding = decimal.ROUND_DOWN

ap = argparse.ArgumentParser()

ap.add_argument("-y", "--yolo", required=False, default="yolo_model",
                help="base path to YOLO directory")

ap.add_argument("-c", "--confidence", type=float, default=0.3,
                help="minimum probability to filter weak detections")

ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "classes.names"])
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "custom.weights"])
configPath = os.path.sep.join([args["yolo"], "custom.cfg"])
print("[INFO] loading YOLO from disk...")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

mts = 0.0000002645833


def listToString(s):
    return ''.join(str(x) for x in s)


def insert(total_area, stain_loc_mts, count, coordinate, total_stain_area):
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="",
        database="fabric"
    )

    mycursor = mydb.cursor()

    sql = "INSERT INTO details (total_length, filename, stain_position_mts, total_stain,xy,area) VALUES " \
          "(%s, %s, %s, %s, %s, %s) "
    val = (
    total_area, os.path.basename(path), listToString(stain_loc_mts), count, listToString(coordinate), total_stain_area)
    mycursor.execute(sql, val)

    mydb.commit()
    row_count = mycursor.rowcount
    if row_count == 0:
        print("Error values not inserted")
    else:
        print("Values inserted successfully")
    mycursor.close()

    mydb.close()


def browse():
    global path
    path = tkinter.filedialog.askopenfilename()
    count = 0
    if len(path) > 0:
        print(path)
        total_area = 0
        image = cv2.imread(path)
        (H, W) = image.shape[:2]

        total_area = H * W * mts
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        boxes = []
        confidences = []
        classIDs = []
        stain_loc_mts = []
        total_stain_area = 0
        coordinate = []
        for output in layerOutputs:

            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > args["confidence"]:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    total_stain_area = total_stain_area + (width * height)

                    stain_loc_mts.append([float(round(float((x * mts)),5)), float(round(float((y * mts)),5))])
                    # stain_loc_mts.append([x * mts, y * mts])
                    coordinate.append([x, y])
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        # print(stain_loc_mts)
        total_stain_area = float(round((total_stain_area * mts),2))

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                args["threshold"])

        if len(idxs) > 0:
            for i in idxs.flatten():
                count += 1

        insert(total_area, stain_loc_mts, count, coordinate, total_stain_area)
        ##################################################################
        hww, www, channels = image.shape

        half = www // 2

        half2 = hww // 2

        top = image[:half2, :]
        bottom = image[half2:, :]

        cv2.imshow('Top', top)
        cv2.imshow('Bottom', bottom)

        cv2.imwrite('crop/top.jpg', top)
        cv2.imwrite('crop/bottom.jpg', bottom)
        cv2.waitKey(0)

        basepath = Path('crop/')
        files_in_basepath = basepath.iterdir()
        for item in files_in_basepath:
            print(item)
            image = cv2.imread(str(item))
            (H, W) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()
            print("[INFO] YOLO took {:.6f} seconds".format(end - start))

            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > args["confidence"]:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                                    args["threshold"])

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

            image = cv2.resize(image, (640, 480))
            cv2.imshow("Image", image)
            cv2.waitKey(0)



root = Tk()

root.geometry("600x300")

label = Label(root, text="Cloth defect detector using YOLO", font=("Courier", 18, "italic", "bold"))

label.place(x=250, y=25, anchor="center")

btn = Button(root, text="Browse", command=browse)
btn.place(x=300, y=150, anchor="center")
mainloop()
