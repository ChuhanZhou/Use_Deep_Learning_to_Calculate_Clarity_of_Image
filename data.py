import numpy as np
import cv2
from config import config as cfg
import torch.nn as nn

def read_data():
    data_set = []
    path_file = open(cfg["train_data"], "r").readlines()
    for path_point in path_file:
        path_point = path_point.split("\n")[0]
        if len(path_point.split(" ")) == 9:
            path = path_point.split(" ")[0]
            img = cv2.imread(path)
            data = img_to_data(img)

            ltPoint = [int(path_point.split(" ")[2]), int(path_point.split(" ")[1])]
            rtPoint = [int(path_point.split(" ")[4]), int(path_point.split(" ")[3])]
            lbPoint = [int(path_point.split(" ")[6]), int(path_point.split(" ")[5])]
            rbPoint = [int(path_point.split(" ")[8]), int(path_point.split(" ")[7])]
            data_set.append([data,point_to_label(np.array([ltPoint, rtPoint, lbPoint, rbPoint]),img.shape)])
    return data_set

def heatmap_to_point(heatmaps, shape):
    cut_h = shape[0]
    cut_w = shape[1]
    input_h = shape[0]
    input_w = shape[1]

    points = []
    for heatmap in heatmaps:
        pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        point0 = cut_h * (pos[0] / input_h)
        point1 = cut_w * (pos[1] / input_w)
        points.append([int(point0), int(point1)])
    return np.array(points)

def point_to_label(points,shape):
    label = np.zeros((len(points),shape[0],shape[1]))
    for i in range(len(points)):
        point = points[i]
        label[i][point[0]][point[1]] = 255
    return label

def img_to_data(img):
    data = np.zeros((2,img.shape[0],img.shape[1]))
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    img_binary = binary_plus(img_gray)
    data[0] = img_gray
    data[1] = img_binary
    return data

def data_to_img(data):
    return data[0],data[1]

def binary(image):
    median_rate = 0.25

    median = np.median(image)
    mean = np.mean(image)
    #print("mean:{} median:{}".format(mean,median))
    if median>mean:
        line = np.mean(image)-np.median(image)*0.15
    else:
        line = np.mean(image)*(1-median_rate) + np.median(image) * median_rate
    ret, thresh1 = cv2.threshold(image, line, 255, cv2.THRESH_BINARY)
    return thresh1

def binary_plus(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #image = cv2.medianBlur(image, 3)
    per = 1/20
    sp = image.shape
    img_for_background = cv2.resize(image,(int(sp[1]*per),int(sp[0]*per)))
    background = calculate_brightness_background(img_for_background,1)
    background = cv2.resize(background,(sp[1],sp[0]))
    after_adjustment = brightness_adjustment(image,background)
    mean = np.mean(after_adjustment)
    line = mean
    ret, img_binary = cv2.threshold(after_adjustment, line, 255, cv2.THRESH_BINARY)
    img_binary = img_binary.astype('float32')
    return img_binary

def show_result_on_img(img, points,p_c = (0,255,0)):
    colors = (0, 0, 0)
    result = img
    if len(img.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    i = 0
    for point in points:
        if point[0] in range(img.shape[0]) and point[1] in range(img.shape[1]) :
            n = 3
            y_s = max(point[0] - n + 1, 0)
            y_e = min(point[0] + n, result.shape[0] - 1)
            x_s = max(point[1] - n + 1, 0)
            x_e = min(point[1] + n, result.shape[1] - 1)

            result[y_s:y_e, x_s:x_e, 0] = p_c[0]
            result[y_s:y_e, x_s:x_e, 1] = p_c[1]
            result[y_s:y_e, x_s:x_e, 2] = p_c[2]
            result[point[0]][point[1]][0] = colors[0]
            result[point[0]][point[1]][1] = colors[1]
            result[point[0]][point[1]][2] = colors[2]
        i += 1
    return result

#base on http://www.javashuo.com/article/p-xxqdgcxk-nr.html
def calculate_brightness_background(img,sampleRadius=1):
    if len(img.shape)==3:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if sampleRadius<=0:
        sampleRadius = 1
    output = np.zeros(img.shape)
    width = img.shape[1]
    height = img.shape[0]

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            sample_start_x = max(x - sampleRadius,0)
            sample_start_y = max(y - sampleRadius,0)
            sample_end_x = min(x + sampleRadius,width-1)
            sample_end_y = min(y + sampleRadius,height-1)
            if sample_end_x == width-1 and sample_end_y == height-1:
                filteringArea = img[sample_start_y:, sample_start_x:]
            elif sample_end_x == width-1:
                filteringArea = img[sample_start_y:sample_end_y + 1, sample_start_x:]
            elif sample_end_y == height-1:
                filteringArea = img[sample_start_y:, sample_start_x:sample_end_x + 1]
            else:
                filteringArea = img[sample_start_y:sample_end_y+1,sample_start_x:sample_end_x+1]
            filteringArea = filteringArea.reshape((filteringArea.shape[0]*filteringArea.shape[1]))
            np.sort(filteringArea).tolist()
            brightness = filteringArea[max(-7,-len(filteringArea)):-1]
            brightness = np.array(brightness).sum()/len(brightness)
            output[y][x] = brightness
    return output

def brightness_adjustment(img, brightness_background=None):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    width = img.shape[1]
    height = img.shape[0]

    output = np.zeros(img.shape)
    if brightness_background is None or brightness_background.shape!=img.shape:

        brightness_background = calculate_brightness_background(img,1)

    for x in range(width):
        for y in range(height):
            background_brightness = brightness_background[y][x]
            original_brightness = img[y][x]
            if background_brightness<=original_brightness:
                after_adjustment = 255
            else:
                b1 = 2.5
                b2 = 1.0
                if background_brightness<20:
                    k = b1
                elif background_brightness<=100:
                    k = 1+(b1-1)*(100-background_brightness)/80
                elif background_brightness<200:
                    k = 1
                else:
                    k = 1+b2*(background_brightness-220)/35
                after_adjustment = int(max(255*0.75,255-k*(background_brightness-original_brightness)))
            output[y][x] = after_adjustment
    return output

def mapping_points(img_o,img_mark_points,points):
    shape_o = img_o.shape
    shape_mp = img_mark_points.shape
    x_pect = shape_o[1]/shape_mp[1]
    y_pect = shape_o[0]/shape_mp[0]
    img_o_points = []
    for point in points:
        x_o = int(point[1]*x_pect)
        y_o = int(point[0]*y_pect)
        img_o_points.append((y_o,x_o))
    return img_o_points

if __name__ == "__main__":
    p = "train/pathPointList_cut_ir.txt"
    infoList = open(p,"r").readlines()
    for info in infoList:
        path = info.split(" ")[0]
        img = cv2.imread(path)
        img = img[0:960,:,:]
        cv2.imwrite(path,img)
