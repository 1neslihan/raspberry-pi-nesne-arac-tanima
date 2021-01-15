import cv2

kamera = cv2.VideoCapture(0)
kamera.set(3,1280)
kamera.set(4,720)
kamera.set(10,70)

classNames= ['insan', 'bisiklet','araba','motorsiklet','ucak','otobus','tren','kamyon','bot','trafik isigi','yangin muslugu', \
'sokak tabelasi','dur isareti','parkmetre','bank','kus','kedi','kopek','at','koyun','inek','fil','ayi','zebra','zurafa','yarasa', \
'sirt cantasi', 'semsiye', 'ayakkabi', 'gozluk', 'el cantasi', 'kravat', 'bavul', 'frizbi', 'kayak', 'snowboard', 'top', 'ucurtma', \
'beyzbol sopasi', 'beyzbol eldiveni', 'kaykay', 'sorf tahtasi', 'tenis raketi', 'sise', 'tabak', 'sarap bardagi', 'kupa', 'catal', \
'bicak', 'kasik', 'kase', 'muz', 'elma', 'sandvic', 'portakal', 'brokoli', 'havuc', 'sosisli', 'pizza', 'donut', 'pasta', 'sandalye', \
'koltuk', 'saksi', 'yatak', 'ayna', 'yemek masasi', 'pencere', 'masa', 'tuvalet', 'kapi', 'tv', 'laptop', 'mouse', 'kumanda', 'klavye', \
'telefon', 'microdalga firin', 'firin', 'tost makinesi', 'lavabo', 'buzdolabi', 'blender', 'kitap', 'saat', 'vazo', 'makas', 'oyuncak ayi',\
'sac kurutma makinesi', 'dis fircasi', 'sac fircasi']

ayarYolu = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
kurallarYolu = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(kurallarYolu,ayarYolu)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = kamera.read()
    classIds, confs, bbox = net.detect(img,confThreshold=0.45)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    cv2.imshow("Sonuclar",img)
    cv2.waitKey(1)
