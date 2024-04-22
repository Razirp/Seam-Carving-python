import os.path

import numpy as np
import cv2
from SeamCarving import SeamCarver

def edgeScharr(srcImage):
    result = cv2.Scharr(srcImage[:2, :, :], -1, 1, 0)[0, :, :]
    return result


def getEnergyMap(tempImage):
    """
    计算出tempImage当前的能量图并将其返回。

    :return: tempImage的当前能量图。
    """
    b, g, r = tempImage[:, :, 0], tempImage[:, :, 1], tempImage[:, :, 2]  # 将图像分为B、G、R三个通道处理
    # 每个通道的能量为cv2.Scharr算子对x方向和y方向分别滤波的绝对值之和
    b_energy = np.absolute(cv2.Scharr(b, -1, 1, 0)) + \
               np.absolute(cv2.Scharr(b, -1, 0, 1))
    g_energy = np.absolute(cv2.Scharr(g, -1, 1, 0)) + \
               np.absolute(cv2.Scharr(g, -1, 0, 1))
    r_energy = np.absolute(cv2.Scharr(r, -1, 1, 0)) + \
               np.absolute(cv2.Scharr(r, -1, 0, 1))
    return b_energy + g_energy + r_energy


def getEnergyMap2(tempImage):
    """
    计算出tempImage当前的能量图并将其返回。

    :return: tempImage的当前能量图。
    """

    energy = np.absolute(cv2.Scharr(tempImage, -1, 1, 0)).sum(2) + \
             np.absolute(cv2.Scharr(tempImage, -1, 0, 1)).sum(2)
    return energy


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    # 输入输出目录。运行前需保证以下两个文件夹在根目录中存在
    folder_in = "in"  # 输入的图片需存放在 ./in 中
    folder_out = "out"  # 输出的图片将存放在 ./out 中

    # 输入文件名和输出文件名。
    imageName_input = "image.jpg"  # 输入的图片的文件名
    imageName_output = ["image_0_re.jpg", "image_0_en.jpg", "image_re_0.jpg",
                        "image_en_0.jpg", "image_en_en.jpg", "image_en_re.jpg", "image_re_en.jpg",
                        "image_re_re.jpg"]  # 输出的图片的文件名

    target_height = [384, 384, 200, 500, 400, 400, 300, 250]  # 缩放后图像的目标高度
    target_width = [300, 800, 512, 512, 800, 400, 700, 350]  # 缩放后图像的目标宽度

    # 输入输出的路径
    inputPath = os.path.join(folder_in, imageName_input)  # 输入图像的路径
    outputPath = os.path.join(folder_out, imageName_output[0])  # 输出图像的路径

    # srcImage = cv2.imread(inputPath).astype(np.double)
    # # print("srcImage:")
    # # print(srcImage[:,:,0])
    # scharr = cv2.Scharr(srcImage, -1, 1, 0)
    # print(scharr.shape)
    # print("scharr:")
    # n = srcImage.shape[0]-1
    # print(scharr[n,:,:])
    # edge = edgeScharr(srcImage)
    # print("edge:")
    # print(edge)

    # print((scharr[0,:,:]==edge).all())
    # print("scharr.sum:")
    # print(scharr.sum(2))
    # energyMap = getEnergyMap(srcImage)
    # print("energyMap:")
    # print(energyMap)
    # energyMap2 = getEnergyMap2(srcImage)
    # print((energyMap == energyMap2).all())

    #
    resizer = SeamCarver(inputPath, target_height[0], target_width[0])
    resizer.saveResultImage(outputPath)

    # for i in range(1, len(target_height)):
    #     resizer.changeTargetSize(target_height[i], target_width[i])
    #     outputPath = os.path.join(folder_out, imageName_output[i])
    #     resizer.saveResultImage(outputPath)

    # srcImage = resizer.getSrcImage()
    # retImage = resizer.getResultImage()
    #
    #
    # cv2.imshow("srcImage", srcImage)
    # cv2.imshow("resultImage", retImage)
    # cv2.waitKey(0)

    # list = []
    # list.append(0)
    # list.append(1)
    # list.append(2)
    #
    # for i in range(3):
    #     print(list.pop(0))
    #
    # a = np.array([5,4,3,2,1, np.inf])
    # # print(a[:-1])
    # b = np.array([np.inf,1,2,3,4,5])
    # # print(np.concatenate((a,b,[np.inf]),axis=0))
    # print(np.amin((a,b),0))
    # print(np.inf + 65535)
    # print(np.where(a >= b))

    # array = np.arange(30).reshape(5,6)
    # array[-1,:] = array[-1,::-1]
    # print(array)
    # seamsIndex = np.zeros((1, array.shape[0]), dtype=np.uint32)
    # seamsIndex[:, -1] = np.argpartition(array[-1, :], 1)[:1]
    # print(seamsIndex)
    # print(array[-1,seamsIndex[:,-1]])

    # image = cv2.imread(inputPath)
    # imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image", image)
    # cv2.imshow("imageGray", imageGray)
    # cv2.waitKey(0)

    # seamsIndex = np.zeros((4, 5), dtype=np.uint32)
    # print(seamsIndex)
    # # m = 10
    # arr = np.zeros((1,5),dtype=np.uint32)
    # print(arr,arr.shape)
    # # print_hi('PyCharm')
    # # img = cv2.imread("image.jpg")
    # # print(img)
    # array = np.arange(27).reshape(3,3,3)
    # print(array)
    # b = array[:,1,:]*10
    # print(b)
    # print(array)
    # b = np.average(array[1,2:4,:],0)
    # print(b)
    # array = np.insert(array,1,b,1)
    # print(array)
    # a,b,c = array[:,:,0],array[:,:,1],array[:,:,2]
    # print(a)
    # print(array[:,-2:])
    # array = array.transpose(1,0,2)
    # print(array)
    # array = array.swapaxes(0,1)
    # print(array)
    # b = array.swapaxes(0,1)
    # print(b)
    # a = np.array([0,1,2])
    # b = np.zeros((3,2,3))
    # for i in range(3):
    #     b[i,:,:] = np.delete(array[i,:,:], a[i], 0)
    # print(b)
    # print(b.shape)
    # print(b)
    # # a=array[1, :]
    # # print(a)
    # # b = array[:,3]
    # # print(b)
    # # print(a+b)

