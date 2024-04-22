import os.path
import cv2
from SeamCarving import SeamCarver

if __name__ == '__main__':
    # 输入输出目录。运行前需保证以下两个文件夹在根目录中存在
    folder_in = "in"  # 输入的图片需存放在 ./in 中
    folder_out = "out"  # 输出的图片将存放在 ./out 中

    # 输入文件名和输出文件名，用户可自定义以下四个字段
    imageName_input = "image.jpg"  # 输入的图片的文件名
    imageName_output = "image_result.jpg"  # 输出的图片的文件名

    target_height = 400  # 缩放后图像的目标高度
    target_width = 400  # 缩放后图像的目标宽度

    # 输入输出的路径
    inputPath = os.path.join(folder_in, imageName_input)  # 输入图像的路径
    outputPath = os.path.join(folder_out, imageName_output)  # 输出图像的路径

    resizer = SeamCarver(inputPath, target_height, target_width)  # 实例化算法类，执行算法
    resizer.saveResultImage(outputPath)  # 保存结果图到目标路径

    srcImage = resizer.getSrcImage()  # 得到原图
    retImage = resizer.getResultImage()  # 得到结果图

    cv2.imshow("Source Image", srcImage)  # 显示原图
    cv2.imshow("Result Image", retImage)  # 显示结果图
    cv2.waitKey(0)
