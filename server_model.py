import cv2
import numpy as np
import time
from pandas import DataFrame
from operator import itemgetter


def load_yolo(params):
    if params == 'detection':
        print(params)
        net = cv2.dnn.readNet("./data/yolov4.weights", "./data/yolov4.cfg")
        classes = []
        with open("./data/classes/coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
    else :
        net = cv2.dnn.readNet("cfg/yolov3-416-clean_5000.weights", "cfg/yolov3-416-clean.cfg")
        classes = []
        with open("./cfg/obj-clean.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layers_names = net.getLayerNames()
        output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

    return net, classes, colors, output_layers


def load_image(img_path):
    # image loading
    print(img_path)
    img = cv2.imread('./' + img_path)
    height, width, channel = img.shape
    print(type(img))
    print("Before resize")
    print(height, width , channel)
    if(height > 500 and width > 500):
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channel = img.shape
    print("After resize")
    print(height, width , channel)
    height, width, channels = img.shape
    return img, height, width, channels # 경로를 통해 불러온 image, 높이 너비 채널수(resize만 됨) - 실사진 크기


def detect_objects(img, net, outputLayers): # load image에서 return된 image가 input으로 -> img에서 binary large objection으로 변환
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    # 416 x 416으로 resize
    # 320 × 320 : 작고 정확도는 떨어지지 만 속도 빠름
    # 608 × 608 : 정확도는 더 높지만 속도 느림
    # 416 × 416 : 중간
    # print(blob[0][0])
    print("blob type : " + str(type(blob)))
    print("blob size : " + str(blob.shape)) # 1, 3, 416, 416
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    confs = []
    class_ids = []
    boxes = []
    for output in outputs:
        # print(output)
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            
            if conf > 0.01:
                # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                
                # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    # print(confs, class_ids, boxes)
    return confs, class_ids, boxes


def draw_labels(params, confs, colors, class_ids, classes, img, boxes):

    print('label size: ', len(class_ids))
    print('score size: ',len(confs))
    result = list()

    indexes = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.4, nms_threshold=0.4)
    
    for i in range(len(boxes)):
        if i in indexes: # 검출된 것 돌면서 라벨링+정확도
            confidence = confs[i]
            label = classes[class_ids[i]]
            color = colors[class_ids[i]]
            x, y, w, h = boxes[i]
            text = f"{label} {confidence:.2f}"
            
            result.append({
                'label' : label,
                'confidence' : confidence
            })
            
            # 사각형 테두리 그리기 및 텍스트 쓰기
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x - 1, y), (x + len(label) * 13 + 65, y - 25), color, -1)
            cv2.putText(img, text, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            
    for key in result:
        print(key)    

    if params == 'detection': # 그 중 중복된 것 지우기.
        result_removed_deduplication = list(
            {result['label']: result for result in result}.values())
        # print("duplicate removed: ", result_removed_deduplication)
        result_sorted = sorted(result_removed_deduplication, key=itemgetter('confidence'), reverse=True)

    else :
        result_sorted = sorted(result, key=itemgetter('confidence'), reverse=True)
        print('sort: ', result_sorted)

    # print('return result: ', result_sorted)
    return result_sorted, img

def image_detect(params, img_path, model, classes, colors, output_layers):
    image, height, width, channels = load_image(img_path) # image 불러와서 image, h w c
    blob, outputs = detect_objects(image, model, output_layers) # 불러온 img -> blob(np.array)로 바꾼 후 net에 돌려서 outputs list출력
    print("Blob : " + str(type(blob)))
    print("outputs : " + str(type(outputs)))
    confs, class_ids, boxes = get_box_dimensions(outputs, height, width)
    result, resultimg = draw_labels(params, confs, colors, class_ids, classes, image, boxes)

    return result, resultimg

def main(imagepath):
    times = []
    params = "detection"
    model, classes, colors, output_layers = load_yolo(params)
    for i in range(0,5):
        timelist = []
        for path in imagepath:
            start = time.time()
            img_path = "./data/"+path
            result, resultimg = image_detect(params, img_path, model, classes, colors, output_layers)
            end = time.time();
            timelist.append(end-start)
            # print(type(resultimg)) # ndarray
            #cv2.imshow("output", resultimg)
        times.append(timelist)
    
    #print(end-start)
    #cv2.waitKey()
    return times
    
if __name__ == '__main__':
    imagepaths = ["car_person.jpg", "sang.jpg", "car.jpg", "kite.jpg", "girl.png"]
    times = main(imagepaths)
    df = DataFrame(times, columns=imagepaths)
    print(df)