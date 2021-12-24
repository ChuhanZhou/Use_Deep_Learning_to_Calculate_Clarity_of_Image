import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from numpy import fft
import matplotlib.pyplot as plt

def FFT(img):
    plt.figure(figsize=(15, 9))
    plt.subplot(231), plt.imshow(img), plt.title('picture')

    # 根据公式转成灰度图
    gray = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
    #img = gray(img)

    # 显示灰度图
    plt.subplot(232), plt.imshow(gray, 'gray'), plt.title('original')

    # 进行傅立叶变换，并显示结果
    fft2 = np.fft.fft2(gray)
    plt.subplot(233), plt.imshow(np.abs(fft2), 'gray'), plt.title('fft2')


    # 将图像变换的原点移动到频域矩形的中心，并显示效果
    shift2center = np.fft.fftshift(fft2)
    plt.subplot(234), plt.imshow(np.abs(shift2center), 'gray'), plt.title('shift2center')

    # 对傅立叶变换的结果进行对数变换，并显示效果
    log_fft2 = np.log(1 + np.abs(fft2))
    plt.subplot(235), plt.imshow(log_fft2, 'gray'), plt.title('log_fft2')

    # 对中心化后的结果进行对数变换，并显示结果
    log_shift2center = np.log(1 + np.abs(shift2center))
    plt.subplot(236), plt.imshow(log_shift2center, 'gray'), plt.title('log_shift2center')
    plt.show()

    return img

def get_trans_list(fourPointListPath):
    transList = []
    fourPointListFile = open(fourPointListPath, "r").readlines()
    for line in fourPointListFile:
        path = line.split(" ")[0]
        if len(line.split(" ")) == 9:
            ltPoint = [int(line.split(" ")[1]), int(line.split(" ")[2])]
            rtPoint = [int(line.split(" ")[3]), int(line.split(" ")[4])]
            lbPoint = [int(line.split(" ")[5]), int(line.split(" ")[6])]
            rbPoint = [int(line.split(" ")[7]), int(line.split(" ")[8])]
            transList.append([path, ltPoint, rtPoint, lbPoint, rbPoint])
    return transList

def four_point_transform(image, ltPoint, rtPoint, lbPoint, rbPoint, pointWidth, pointHeight):
    rect = np.array([ltPoint, rtPoint, rbPoint, lbPoint], dtype="float32")

    # topWidth = np.sqrt(((ltPoint[0] - rtPoint[0]) ** 2) + ((ltPoint[1] - rtPoint[1]) ** 2))
    # bottomWidth = np.sqrt(((lbPoint[0] - rbPoint[0]) ** 2) + ((lbPoint[1] - rbPoint[1]) ** 2))
    # maxWidth = max(int(topWidth), int(bottomWidth))

    # leftHeight = np.sqrt(((ltPoint[0] - lbPoint[0]) ** 2) + ((ltPoint[1] - lbPoint[1]) ** 2))
    # rightHeight = np.sqrt(((rtPoint[0] - rbPoint[0]) ** 2) + ((rtPoint[1] - rbPoint[1]) ** 2))
    # maxHeight = max(int(leftHeight), int(rightHeight))

    dst = np.array([
        [pointWidth * 1.5 - 1, pointHeight * 0.5 - 1],
        [pointWidth * 2.5 - 1, pointHeight * 0.5 - 1],
        [pointWidth * 2.5 - 1, pointHeight * 1.5 - 1],
        [pointWidth * 1.5 - 1, pointHeight * 1.5 - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    result = cv2.warpPerspective(image, M, (pointWidth * 4, pointHeight * 2))
    return result

def point_transform(image1_info,image2_info):
    path1 = image1_info.split(" ")[0]
    point_list1 = []
    image1 = cv2.imread(path1)
    for i in range(int((len(image1_info.split(" "))-1)/2)):
        p_x = image1_info.split(" ")[i*2+1]
        p_y = image1_info.split(" ")[i*2+2]
        point_list1.append([int(p_x),int(p_y)])

    path2 = image2_info.split(" ")[0]
    point_list2 = []
    image2 = cv2.imread(path2)
    for i in range(int((len(image2_info.split(" ")) - 1) / 2)):
        p_x = image2_info.split(" ")[i * 2 + 1]
        p_y = image2_info.split(" ")[i * 2 + 2]
        point_list2.append([int(p_x),int(p_y)])

    rect = np.array(point_list2, dtype="float32")

    dst = np.array(point_list1, dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    result = cv2.warpPerspective(image2, M, (image1.shape[1],image1.shape[0]))
    return result,image1

# 峰值信噪比:越高失真越小
def psnr(image1, image2):
    return compare_psnr(image1, image2)

# 结构相似性:[0,1]越高失真越小
def ssim(image1, image2):
    ssim = compare_ssim(image1, image2, multichannel=True)
    return ssim

def mse(image1, image2):
    mse = compare_mse(image1, image2)
    return mse

def get_check_part(image):
    sp = image.shape
    height = sp[0]
    width = sp[1]

    leftPart1 = image[int(277 / 900 * height):int(750 / 900 * height),int(120 / 800 * width):int(264 / 800 * width)]
    leftPart10 = image[int(277 / 900 * height):int(583 / 900 * height), int(120 / 800 * width):int(264 / 800 * width)]
    leftPart11 = image[int(277 / 900 * height):int(395 / 900 * height), int(120 / 800 * width):int(264 / 800 * width)]
    leftPart12 = image[int(395 / 900 * height):int(583 / 900 * height), int(120 / 800 * width):int(264 / 800 * width)]
    leftPart2 = image[int(800 / 900 * height):int(855 / 900 * height),int(0 / 800 * width):int(318 / 800 * width)]
    middlePart1 = image[int(163 / 900 * height):int(735 / 900 * height),int(262 / 800 * width):int(536 / 800 * width)]
    middlePart2 = image[int(749 / 900 * height):int(899 / 900 * height),int(318 / 800 * width):int(479 / 800 * width)]
    rightPart = image[int(267 / 900 * height):int(770 / 900 * height),int(536 / 800 * width):int(629 / 800 * width)]
    part1 = image[int(460 / 900 * height):int(560 / 900 * height), int(210 / 800 * width):int(250 / 800 * width)]
    part2 = image[int(267 / 900 * height):int(770 / 900 * height), int(536 / 800 * width):int(629 / 800 * width)]
    #return [image,leftPart11,leftPart12,leftPart2,middlePart1,middlePart2,rightPart]
    return [image]

def energy(image):
    image = gray(image)
    global img
    if image.ndim == 3:
        img = image[:, :, 0]
    shape = np.shape(img)
    out = 0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            out+=((int(img[x+1,y])-int(img[x,y]))**2)+((int(img[x,y+1]-int(img[x,y])))**2)
    return out

def laplacian(image):
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return variance_of_laplacian(gray)

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def gray(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayImage = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2BGR)
    return grayImage

def run(fourPointListPath, outputPackagePath,originalName="ISO_12233-reschart-1.png",need_flip=True):
    transList = get_trans_list(fourPointListPath)
    outputList = []
    original = []
    #clarityModel = energy.__name__
    clarityModel = laplacian.__name__
    for transInfo in transList:
        image = cv2.imread(transInfo[0])
        pointLength = 12 #对齐roi比例
        after = four_point_transform(image, transInfo[1], transInfo[2], transInfo[3], transInfo[4], 20 * pointLength,45 * pointLength)
        outputPath = "{}/{}".format(outputPackagePath, transInfo[0].split("/")[-1].split(".")[0])

        if transInfo[0].split("/")[-1] != originalName:
            if need_flip:
                after = cv2.flip(after, 1)
            outputList.append([transInfo[0].split("/")[-1].split(".")[0], get_check_part(after)])
        else:
            original = get_check_part(after)
        cv2.imwrite("{}.png".format(outputPath), after, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        parts = get_check_part(after)
        index = 0
        for part in parts:
            cv2.imwrite("{}_{}.png".format(outputPath,index), part, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            index+=1
        print("output:{}".format(outputPath))

    originalClarity = []
    for part in original:
        if clarityModel == "energy":
            originalClarity.append(energy(part))
        elif clarityModel == "laplacian":
            originalClarity.append(laplacian(part))

    for outputInfo in outputList:
        name = outputInfo[0]
        clarityList = []
        psnrList = []
        ssimList = []
        mseList = []

        for index in range(len(outputInfo[1])):
            if clarityModel == "energy":
                clarityList.append(energy(outputInfo[1][index])/originalClarity[index])
            elif clarityModel == "laplacian":
                clarityList.append(laplacian(outputInfo[1][index])/originalClarity[index])
            #psnrList.append(psnr(outputInfo[1][index],original[index]))
            #ssimList.append(ssim(outputInfo[1][index],original[index]))
            #mseList.append(mse(outputInfo[1][index],original[index]))

        clarityInfo = "{} clarity({}):".format(name,clarityModel)
        for clarity in clarityList:
            clarityInfo += "{:.6f} ".format(clarity)

        pp = "psnr:"
        for p in psnrList:
            pp += "{:.6f} ".format(p)
        sp = "ssim:"
        for s in ssimList:
            sp += "{:.6f} ".format(s)
        mp = "mse:"
        for m in mseList:
            mp += "{:.6f} ".format(m)

        print(clarityInfo)

if __name__ == "__main__":
    run("test/path_points_list.txt","result",need_flip=False)
