#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#載入必要模組
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2

#使用參數方式傳入Training和Test的dataset
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())

#data用來存放HOG資訊，labels則是存放對應的標籤
data = []
labels = []

#依序讀取training dataset中的圖檔
for imagePath in paths.list_images(args["training"]):
        #將資料夾的名稱取出作為該圖檔的標籤
        make = imagePath.split("/")[-2]

        #----以下為訓練圖檔的預處理----# 
        #載入圖檔,轉為灰階,作模糊化處理
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            #作threshold(固定閾值)處理
        (T, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
            #使用Canny方法偵側邊緣
        edged = imutils.auto_canny(thresh)
        #尋找輪廓，只取最大的那個
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)

        #取出輪廓的長寬高，用來裁切原圖檔。
        (x, y, w, h) = cv2.boundingRect(c)
        Cutted = gray[y:y + h, x:x + w]
        #將裁切後的圖檔尺寸更改為500x500。
        Cutted = cv2.resize(Cutted, (500, 500))
        cv2.imshow("Cutted", Cutted)
#----訓練圖檔預處理結束----#
        #取得其HOG資訊及視覺化圖檔
        (H, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10),
                cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
        #將HOG資訊及標籤分別放入data及labels陣列
        data.append(H)
        labels.append(make)

#開始用KNN模型來訓練
model = KNeighborsClassifier(n_neighbors=1)
#傳入data及labels陣列開始訓練
model.fit(data, labels)

#準備使用Test Dataset來驗証
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
        #從測試資料中讀取圖檔
        image = cv2.imread(imagePath)
        #----以下為測試圖檔的預處理----# 
        #轉為灰階並模糊化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            #作threshold(固定閾值)處理
        (T, thresh) = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
            #使用Canny方法偵側邊緣
        edged = imutils.auto_canny(thresh)
        #尋找輪廓，只取最大的那個
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)

        #取出輪廓的長寬高，用來裁切原圖檔。
        (x, y, w, h) = cv2.boundingRect(c)
        Cutted = gray[y:y + h, x:x + w]
        #將裁切後的圖檔尺寸更改為500x500。
        Cutted = cv2.resize(Cutted, (500, 500))
        cv2.imshow("Cutted", Cutted)
#----訓練圖檔預處理結束----#

        #取得其HOG資訊及視覺化圖檔
        (H, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10),
                cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
        #使用訓練的模型預測此圖檔
        pred = model.predict(H.reshape(1, -1))[0]

        #顯示HOG視覺化圖檔
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        cv2.imshow("HOG Image #{}".format(i + 1), hogImage)

        #將預測數字顯示在圖片上
        cv2.putText(image, pred.title(), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 3)
        cv2.imshow("Test Image #{}".format(i + 1), image)
        cv2.waitKey(0)

