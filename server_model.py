import cv2
import numpy as np
from keras.models import load_model
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
    print("Before resize")
    print(height, width , channel)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    #img = cv2.resize(img, (640, 480))
    height, width, channel = img.shape
    print("After resize")
    print(height, width , channel)
    height, width, channels = img.shape
    return img, height, width, channels # 경로를 통해 불러온 image, 높이 너비 채널수(resize만 됨) - 실사진 크기


def detect_objects(img, net, outputLayers): # load image에서 return된 image가 input으로 -> img에서 binary large objection으로 변환
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    # 416 x 416으로 resize
    # 320 × 320 : 작고 정확도는 떨어지지 만 속도 빠름
    # 609 × 609 : 정확도는 더 높지만 속도 느림
    # 416 × 416 : 중간
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
                #print(detect)
                
                # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
                print(x, y, w, h)
    #print(confs, class_ids, boxes)
    return confs, class_ids, boxes

def set_class_size(): # distance를 위한 클래스들 (논문 보고 다시 정리하기)
    classes = ['person', 'bus', 'truck', 'car', 'bicycle', 'motorbike', 'cat', 'dog', 'horse', 'sheep', 'cow']
    class_shape = [[175, 55, 30], [300, 250, 1200], [400, 350, 1500], [160, 180, 400], [110, 50, 180],
                [110, 50, 180], [40, 20, 50], [50, 30, 60], [180, 60, 200], [130, 60, 150], [170, 70, 200]]
    set_class_size = dict(zip(classes, class_shape))
    print("Load Class size!")
    return set_class_size

def load_dist_input(predict_box, predict_class, img_width, img_height):
    top, left, box_width, box_height = predict_box
    
    # width = float(right - left) / img_width
    # height = float(bottom - top) / img_height
    width = box_width / img_width
    height = box_height / img_height
    diagonal = np.sqrt(np.square(width) + np.square(height))
    set_class_size_result = set_class_size()
    class_h, class_w, class_d = np.array(set_class_size_result[predict_class], dtype=np.float32)
    dist_input = [1 / width, 1 / height, 1 / diagonal, class_h, class_w, class_d]
    return np.array(dist_input) # numpy 리턴 후 모델에 넣자.

def draw_labels(params, confs, colors, class_ids, classes, img, boxes, distance_model):

    print('label size: ', len(class_ids))
    print('score size: ',len(confs))
    result = list()

    indexes = cv2.dnn.NMSBoxes(boxes, confs, score_threshold=0.4, nms_threshold=0.4)
    
    height, width, channels = img.shape
    
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
            ##### distance model
            distance_input = load_dist_input(boxes[i], label, width, height)
            distance = distance_model.predict(np.array([distance_input]).reshape(-1, 6))
            #####
            print(distance)
            # 사각형 테두리 그리기 및 텍스트 쓰기
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(img, (x - 1, y), (x + len(label) * 13 + 65, y - 25), color, -1)
            #cv2.putText(img, text, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            cv2.putText(img, str(distance), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
            
        print('result: ',result)

    if params == 'detection': # 그 중 중복된 것 지우기.
        result_removed_deduplication = list(
            {result['label']: result for result in result}.values())
        print("duplicate removed: ", result_removed_deduplication)
        result_sorted = sorted(result_removed_deduplication, key=itemgetter('confidence'), reverse=True)

    else :
        result_sorted = sorted(result, key=itemgetter('confidence'), reverse=True)
        print('sort: ', result_sorted)

    print('return result: ', result_sorted)
    return result_sorted, img

def image_detect(params, img_path):
    model, classes, colors, output_layers = load_yolo(params)
    distance_model = load_model("./data/model_3.keras", compile=False)
    image, height, width, channels = load_image(img_path) # image 불러와서 image, h w c
    blob, outputs = detect_objects(image, model, output_layers) # 불러온 img -> blob(np.array)로 바꾼 후 net에 돌려서 outputs list출력
    print("Blob : " + str(type(blob)))
    print("outputs : " + str(type(outputs)))
    print(height, width)
    confs, class_ids, boxes = get_box_dimensions(outputs, height, width)
    result, resultimg = draw_labels(params, confs, colors, class_ids, classes, image, boxes, distance_model)

    return result, resultimg

def main():
    params = "detection"
    img_path = "./data/temp4.jpg"
    result, resultimg = image_detect(params, img_path)
    print(type(resultimg)) # ndarray
    cv2.imshow("output", resultimg)
    cv2.waitKey()
    
if __name__ == '__main__':
    main()