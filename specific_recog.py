import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import math 

c = 0


def showimage(img):
    screen_res = 1280, 720
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)
    # print("working")
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', window_width, window_height)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, c, view_object):
    label = str(classes[class_id])

    color = COLORS[class_id]
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    if label == str(view_object):
        crop_img = img[y:y+h, x:x+w]
        c+=1
        cv2.imwrite('object'+str(c)+'.jpg', crop_img)
        print("created an image for obj"+str(c))
        return 1
    else:
        return 0

def find_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    maxcolor = []
    maxrange = 0
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("int8").tolist(), -1)

        color = np.array(color, dtype = 'int')
        color = np.expand_dims(color, axis=0)
        diff = endX - startX
        if maxrange < diff:
            maxrange = diff
            maxcolor = color
        startX = endX
    print(maxcolor)
    return maxcolor



def color_similarity(query_color, detected_color):
    # print("yo")
    rmean = (query_color[0][0] + detected_color[0][0]) / 2
    r = query_color[0][0] - detected_color[0][0]
    g = query_color[0][1] - detected_color[0][1]
    b = query_color[0][2] - detected_color[0][2]

    weightr = 2 + rmean/256
    weightg = 4
    weightb = 2 + (255- rmean)/256

    dist = math.sqrt(weightr*r*r + weightg*g*g + weightb*b*b)
    # print(dist, "dist")
    if (dist < 50):
        return 1
    else:
        return 0


def checkSimilar(obj, view_image):
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(obj,None)
    kp2, des2 = sift.detectAndCompute(view_image,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    print(len(good), "Number of Key-point matches.")
    if(len(good) > 12):
        return 1
    else:
        return 0

    
image = cv2.imread('./images/occ_lap.jpg')

print("Loaded Test image")
print()


Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

classes = None

with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4


for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])


indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


number = 0


with open('detected_view_object.txt', "r") as f:
    view_object = f.read()
# print("C initialised", c)
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    number += draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), number, view_object)

# cv2.imshow("object detection", image)

# showimage(image)
# cv2.waitKey()
    
# cv2.imwrite("object-detection.jpg", image)
# cv2.destroyAllWindows()



print(number, "is the number of objects identified in the test image which is same as the query object")
print()



img1 = cv2.imread('./images/image_yolo1.jpg')
img2 = cv2.imread('./images/image_yolo2.jpg')
img3 = cv2.imread('./images/image_yolo3.jpg')

print("loaded the crop fitted images of different views of the query object.")
print()

print("Detect the dominant color for each of the view image")

view_images = [img1, img2, img3]
view_img_colors = []

for image in view_images:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = image.reshape((image.shape[0] * image.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)
    hist = find_histogram(clt)
    view_img_colors.append(plot_colors2(hist, clt.cluster_centers_))

print("done detecting dominant color for view images")

# print(view_img_colors)
print()
print()
print()
matched_objects = []


print("Detect the dominant color for each of the cropped image from the test image'")
for i in range(1, number+1):
    image = cv2.imread("object"+str(i)+".jpg")
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    hist = find_histogram(clt)
    detected_color = plot_colors2(hist, clt.cluster_centers_)
    f = 0
    for query_color in view_img_colors:
        f = color_similarity(query_color, detected_color)
        if (f):
            print("True, This object which was cropped from test image has similar color as that of the query object.")
            matched_objects.append(image)
            break
    if (f == 0):
        print("False")


print ("Done detecting the dominant color for objects that are cropped from the test image.")

if (len(matched_objects) == 0):
    print("None of the objects which are cropped from the test image passed the cololor test.")
    print("The object is not in the image.")
    exit(0)
        # print(f)

print()
print("Final Test")

print("Number of objects that passed the Color test", len(matched_objects))
print("Number of view images", len(view_images))


print()
print()
print()

br = 0
final = 0
for obj in matched_objects:
    # i = 0
    for view_image in view_images:
        final = checkSimilar(obj, view_image)
        # cv2.imwrite("obj.jpg", obj)
        # cv2.imwrite("view_image.jpg", view_image)
        if final:
            br = 1
            break
        else:
            print("failed to pass the object similarity test.")
    if (br):
        print("Object Similiarity test passed.")
        print("The object is in the image.")
        break
if (final == 0):
    print("The object is not in the image.")




    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

    # plt.imshow(img3),plt.show()

