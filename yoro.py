import io
import cv2
import torch
import PySimpleGUI as sg
from PIL import Image, ImageTk
import numpy as np

# YOLOv5モデルのロード
yoro_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def process_yoro(frame):
    img = Image.fromarray(frame)
    results = yoro_model(img)
    
    # 人間のクラスIDを取得（YOLOv5の場合、0が人間）
    person_class_id = 0
    
    # フレームを透明化
    frame_alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    frame_alpha[..., 3] = 0  # 全部透明にする

    n = 0
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == person_class_id:
            x1, y1, x2, y2 = map(int, box)
            frame_alpha[y1:y2, x1:x2, 3] = 255  # 人物部分だけ不透明にする

            img_save = np.zeros((y2-y1,x2-x1,3))
            img_save = frame[y1:y2, x1:x2]

        cv2.imwrite("img/frame_alpha_%d.png"%n,img_save)
        n = n + 1

    return frame_alpha

img = cv2.imread("1.jpg")

# PySimpleGUIの設定
sg.theme('Reddit')
layout = [[sg.Image(filename='', key='-IMAGE-')]]
window = sg.Window('Image', layout, location=(800,400))

while True:
    img = cv2.resize(img, (800, 600))
    
    # YOLOv5で人物以外の領域を透明化
    img_alpha = process_yoro(img)

    is_success, buffer = cv2.imencode(".png", img_alpha)
    if is_success:
        # エンコードした画像をPIL.Imageに変換
        bio = io.BytesIO(buffer)
        image = Image.open(bio)

        # PySimpleGUIで表示
        event, values = window.read(timeout=50)
        if event == sg.WINDOW_CLOSED:
            break
        window['-IMAGE-'].update(data=ImageTk.PhotoImage(image))

    if event == sg.WINDOW_CLOSED:
        break
window.close()
