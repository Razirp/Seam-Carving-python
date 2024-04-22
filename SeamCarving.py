import sys
import numpy as np
import cv2
from tqdm import tqdm
import time


class SeamCarver:
    def __init__(self, inputPath, targetHeight, targetWidth):
        """
        初始化字段，得到目标图像。

        Parameters:
        ----------
        :param inputPath: 输入图像的存放路径
        :param targetHeight: 目标高度
        :param targetWidth: 目标宽度
        """
        self.srcImage = cv2.imread(inputPath).astype(np.double)  # 获得输入的源图像
        if self.srcImage is None:
            print("读取图像失败")
            exit(1)
        else:
            print("已读入图像" + inputPath)

        # 初始化字段
        self.targetHeight = targetHeight
        self.targetWidth = targetWidth
        self.tempImage = np.copy(self.srcImage)  # 算法的操作将在复制出的tempImage上进行

        self.energyMap = None  # 初始化能量图为None

        self.start()  # 进行算法操作

    def changeTargetSize(self, targetHeight, targetWidth):
        """
        重新设置目标大小，并重新对图像进行缩放。

        :param targetHeight: 新的目标高度。
        :param targetWidth: 新的目标宽度。
        :return: 没有返回值。
        """
        # 更新字段
        self.targetHeight = targetHeight
        self.targetWidth = targetWidth
        self.tempImage = np.copy(self.srcImage)  # 还原tempImage为原图

        self.energyMap = None  # 能量图重置为None

        self.start()  # 重新执行算法操作

    def start(self):
        """
        Seam Carving 算法的开始部分。
        根据图像尺寸的变化要求选择不同的操作步骤：
        当只有一个方向的尺寸发生变化时，选择一维操作（行方向或列方向），否则选择二维操作。
        - 列方向指处理的接缝为纵向的
        - 行方向指处理的接缝为横向的

        :return: 没有返回值。
        """
        print("算法操作开始...执行变化：")
        print("高×宽：", self.srcImage.shape[0], "×", self.srcImage.shape[1], "==>", self.targetHeight, "×",
              self.targetWidth)
        startTime = time.time()
        delta_rows, delta_cols = int(self.targetHeight - self.srcImage.shape[0]), \
                                 int(self.targetWidth - self.srcImage.shape[1])

        if delta_cols == delta_rows == 0:  # 若没有变化，则无需操作
            pass
        elif delta_cols == 0:  # 若列方向变化为 0，则只需要处理行方向
            self.rowsSeamCarve(delta_rows)
        elif delta_rows == 0:  # 若行方向变化为 0，则只需要处理列方向
            self.colsSeamCarve(delta_cols)
        else:  # 否则需同时处理两个方向
            self.twoDimSeamCarve(delta_rows, delta_cols)
        endTime = time.time()
        print("处理完毕，处理时间共：%.4f秒" % (endTime - startTime))
        print("-----------------------------------------------------------")

    def colsSeamCarve(self, delta_cols):
        """
        仅列数发生变化的情况。将在tempImage中增加delta_cols（若为负则减少）条列方向接缝。

        :param delta_cols: 需要增加的列方向接缝的数量。允许为负，代表减少接缝。
        :return: 没有返回值。
        """
        if delta_cols < 0:  # 列数变化小于零，说明列数减少，需要移除相应数量的列接缝
            self.removeColSeams(-delta_cols)
        elif delta_cols > 0:  # 列数变化大于零，说明列数增加，需要插入相应数量的列接缝
            self.insertColSeams(delta_cols)

    def rowsSeamCarve(self, delta_rows):
        """
        仅行数发生变化的情况。将在tempImage中增加delta_rows（若为负则减少）条行方向接缝。

        - 通过将图像转置然后调用colsSeamCarve()解决。

        :param delta_rows: 需要增加的行方向接缝的数量。允许为负，代表减少接缝。
        :return: 没有返回值。
        """
        self.energyMap = None  # 清空能量图信息，以使能量图得以被重新计算
        self.tempImage = self.tempImage.swapaxes(0, 1)  # 转置图像
        self.colsSeamCarve(delta_rows)  # 在转置图中操作列，实现在原图中操作行的效果
        self.tempImage = self.tempImage.swapaxes(0, 1)  # 将图像转置回来
        self.energyMap = None  # 清空能量图信息，以使能量图得以被重新计算

    def twoDimSeamCarve(self, delta_rows, delta_cols):
        """
        两方向同时发生变化的情况，先处理列变化再处理行变化。

        :param delta_rows: 需要增加的行方向接缝的数量。允许为负，代表减少接缝。
        :param delta_cols: 需要增加的列方向接缝的数量。允许为负，代表减少接缝。
        :return: 没有返回值
        """
        self.colsSeamCarve(delta_cols)
        self.rowsSeamCarve(delta_rows)

    def removeColSeams(self, num_seamsRemoved):
        """
        在tempImage中移除num_seamsRemoved条列方向（竖直）接缝。

        :param num_seamsRemoved: 需要被移除的接缝数量。
        :return: 没有返回值。
        """
        print("正在执行removeSeams()...将移除", num_seamsRemoved, "条接缝：")
        for i in tqdm(range(num_seamsRemoved), file=sys.stdout):  # 循环移除接缝
            self.checkEnergyMap()  # 检查能量图，确保能量图有效。
            seamToRemove = self.get_seamToRemove_col()  # 得到将要移除的接缝（总能量最小的）
            self.doColSeamRemoval(seamToRemove)  # 从tempImage中移除相应接缝

    def insertColSeams(self, num_seamsInserted):
        """
        判断是否需要多次插入：
        - 当扩大的比例超过原尺寸的p倍时，将分多次插入，每次插入的规模是之前的0.8倍，直到达到要求。

        :param num_seamsInserted: 需要插入的接缝数量。
        :return: 没有返回值。
        """
        p = 0.3
        limit = int(p * self.tempImage.shape[1])  # 限制一次最多插入的接缝数
        insertedNum = 0
        print("正在执行insertSeams()...将插入", num_seamsInserted, "条接缝：")
        count = 0
        while insertedNum < num_seamsInserted:
            count += 1
            curInsertNum = min(limit, num_seamsInserted - insertedNum)  # 计算本次需要插入的条数
            print("第", count, "次插入，将插入", curInsertNum, "条接缝：")
            self.insertColSeams_sub(curInsertNum)  # 插入
            limit = max(1, int(limit * 0.8))  # 更新限制
            insertedNum += curInsertNum

    def insertColSeams_sub(self, num_seamsInserted):
        """
        在tempImage中插入num_seamsInserted条列方向（竖直）接缝。

        :param num_seamsInserted: 需要插入的接缝数量。
        :return: 没有返回值。
        """
        seamsQueue = []  # 存放所需插入的接缝
        copiedImage = np.copy(self.tempImage)  # 暂存tempImage的初始状态

        # 通过依次删除num_seamsInserted条接缝，找到总能量前num_seamsInserted小的接缝存于seamsQueue中

        print("正在定位接缝位置：")
        for i in tqdm(range(num_seamsInserted), file=sys.stdout):
            self.checkEnergyMap()  # 检查能量图，确保能量图有效。
            seamToRemove = self.get_seamToRemove_col()  # 得到将要移除的接缝（总能量最小的）
            seamsQueue.append(seamToRemove)  # 记录下相应的接缝
            self.doColSeamRemoval(seamToRemove)  # 从tempImage中移除相应接缝

        self.tempImage = np.copy(copiedImage)  # 恢复tempImage为初始状态
        self.energyMap = None  # 清空能量图，指示下一次需重新计算能量图。

        print("正在插入接缝：")
        # 依照seamsQueue中记录的接缝位置依次将接缝插入到图中，并实时更新seamsQueue
        for i in tqdm(range(num_seamsInserted), file=sys.stdout):
            seamToInsert = seamsQueue.pop(0)  # 取队列头对应的接缝进行插入
            self.doColSeamInsertion(seamToInsert)  # 插入接缝
            seamsQueue = self.updateSeamsQueue(seamsQueue, seamToInsert)  # 更新seamsQueue

    def updateSeamsQueue(self, seamsQueue, lastInsertedSeam):
        """
        根据最近被插入的接缝更新seamsQueue中的坐标并返回。

        :param seamsQueue: 需要被更新的seamsQueue，存储了一系列接缝索引。
        :param lastInsertedSeam: 上一个被插入的接缝的索引。
        :return: 新的seamsQueue。
        """
        newSeamsQueue = []
        for seam in seamsQueue:  # 遍历更新每一个seam
            seam[np.where(lastInsertedSeam <= seam)] += 2  # 当上一个接缝插入到seam左侧时，将seam自增2（包括新插入的像素和其对应的原图中的像素）
            newSeamsQueue.append(seam)
        return newSeamsQueue

    def checkEnergyMap(self):
        """
        检查能量图是否可用，若不可用则重新计算能量图。

        :return: 没有返回值。
        """
        if self.energyMap is None:  # 如果没有计算过梯度图，则进行初始化计算
            self.energyMap = np.absolute(cv2.Scharr(self.tempImage, -1, 1, 0)).sum(2) + \
                             np.absolute(cv2.Scharr(self.tempImage, -1, 0, 1)).sum(2)

    def get_seamToRemove_col(self):
        """
        返回应当被移除的列（竖直）接缝。

        :return: 列表，指示应当被移除的列接缝在每一行的索引。
        """
        dpMap = self.getDPMap_colSeams()
        return self.findOptimalSeam_col(dpMap)

    def getDPMap_colSeams(self):
        """
        得到计算纵向接缝的动态规划图。

        :return: 计算纵向接缝的动态规划图。
        """
        dpMap = np.copy(self.energyMap)  # 动态规划图与能量图同形，在能量图基准上累加
        # 从第二行遍历到最后一行
        for row in range(1, self.energyMap.shape[0]):
            left = np.concatenate((dpMap[row - 1, 1:], [np.inf]), axis=0)  # 上一行左移
            right = np.concatenate(([np.inf], dpMap[row - 1, :self.energyMap.shape[1] - 1]), axis=0)  # 上一行右移
            dpMap[row, :] += np.amin((left, dpMap[row - 1, :], right), axis=0)  # 取上一行左中右三个值中的最小值与当前行累加
            # for col in range(energyMap.shape[1]):
            #     dpMap[row, col] += \
            #         np.amin(dpMap[row - 1, max(0, col - 1):
            #                                min(energyMap.shape[1], col + 2)])  # 注意“:”右边取不到
        return dpMap

    def findOptimalSeam_col(self, dpMap):
        """
        由dpMap回溯得到总能量最小的纵向接缝的索引列表。

        :param dpMap: 计算接缝得到的动态规划图。
        :return: 总能量最小的纵向接缝的索引列表。
        """
        seamIndex = np.zeros((dpMap.shape[0],), dtype=np.uint32)  # 初始化接缝的索引列表
        seamIndex[-1] = np.argmin(dpMap[-1, :])  # 取最小总能量对应的索引开始回溯
        for row in range(dpMap.shape[0] - 2, -1, -1):  # 从倒数第二行开始倒序遍历到第一行
            preIndex = seamIndex[row + 1]
            seamIndex[row] = max(preIndex - 1, 0) + \
                             np.argmin(dpMap[row,
                                       max(preIndex - 1, 0):
                                       min(preIndex + 2, dpMap.shape[1])])  # 注意“:”右边取不到
        return seamIndex

    def doColSeamRemoval(self, seamToRemove):
        """
        从tempImage中移除seamsToRemove所指示的竖直接缝。

        :param seamToRemove: 需要移除的接缝的索引列表。
        :return: 没有返回值。
        """
        newImage = np.zeros((self.tempImage.shape[0],
                             self.tempImage.shape[1] - 1,
                             self.tempImage.shape[2]))
        for row in range(self.tempImage.shape[0]):  # 遍历更改每一行
            newImage[row, :, :] = np.delete(self.tempImage[row, :, :], seamToRemove[row], 0)  # 移除相应索引处的像素
        self.tempImage = np.copy(newImage)  # 更新图像
        self.updateEnergyMap(seamToRemove)  # 更新能量图

    def doColSeamInsertion(self, seamToInsert):
        """
        在seamToInsert处插入一条新的接缝，像素值取左右像素值的平均值。

        :param seamToInsert: 需要插入的接缝的索引列表。
        :return: 没有返回值
        """
        newImage = np.zeros((self.tempImage.shape[0],
                             self.tempImage.shape[1] + 1,  # 增加1列
                             self.tempImage.shape[2]))
        for row in range(self.tempImage.shape[0]):  # 遍历更改每一行
            curCol = seamToInsert[row]  # 当前需操作的列
            newPixel = np.average(self.tempImage[row, curCol:curCol + 2, :], 0)  # 新像素值取seam和其后一个像素的平均值
            newImage[row, :, :] = np.insert(self.tempImage[row, :, :], curCol, newPixel, 0)  # 在相应位置插入新像素
        self.tempImage = np.copy(newImage)  # 更新图像

    def updateEnergyMap(self, removedSeam):
        """
        在每次移除接缝之后根据图像变化更新能量图。
        - 利用 cv2.Scharr 算子对接缝左右边界（各向外扩大一格）之间的图像区域重新进行滤波。

        :param removedSeam: 被移除的接缝。指示图像发生的变化。
        :return: 没有返回值。
        """
        colLimit = self.tempImage.shape[1]  # 图像列边界
        leftLimit, rightLimit = max(0, np.amin(removedSeam) - 1), \
                                min(colLimit, np.amax(removedSeam) + 1)  # 需要重新滤波的区域的边界（左闭右开），因为接缝被删除，所以右边界无需再扩大。
        self.energyMap = np.delete(self.energyMap, leftLimit, 1)  # 从energyMap中移除一列，实现更新能量图矩阵形状的效果
        scharrLeftLimit, \
        scharrRightLimit = max(0, leftLimit - 1), \
                           min(colLimit, rightLimit + 1)  # 滤波算子需要读入的区域的边界。通常比需要重新滤波的边界左右各宽1个单位。
        tempMap = np.absolute(cv2.Scharr(self.tempImage[:, scharrLeftLimit:scharrRightLimit, :], -1, 1, 0)).sum(2) + \
                  np.absolute(cv2.Scharr(self.tempImage[:, scharrLeftLimit:scharrRightLimit, :], -1, 0, 1)).sum(2)  # 滤波算子结果
        # 根据情况裁剪滤波算子结果得到有效部分更新到能量图中
        if leftLimit == 0 and rightLimit == colLimit:
            self.energyMap[:, leftLimit:rightLimit] = np.copy(tempMap)
        elif leftLimit == 0:  # 左边界到头，只需将最后一列裁去
            self.energyMap[:, leftLimit:rightLimit] = tempMap[:, :-1]
        elif rightLimit == colLimit:  # 右边界到头，只需将第一列裁去
            self.energyMap[:, leftLimit:rightLimit] = tempMap[:, 1:]
        else:  # 一般情况下需要左右各裁去一列
            self.energyMap[:, leftLimit:rightLimit] = tempMap[:, 1:-1]

    def getResultImage(self):
        """
        将结果图以cv2::BGR格式返回。

        :return: cv2::BGR格式的结果图。
        """
        return self.tempImage.astype(np.uint8)

    def saveResultImage(self, outputPath):
        """
        将结果图存储到输出路径。

        :param outputPath: 结果图存储路径
        :return: 没有返回值。
        """
        cv2.imwrite(outputPath, self.tempImage.astype(np.uint8))
        print("图像已经存储到：", outputPath)
        print("-----------------------------------------------------------")

    def getSrcImage(self):
        """
        将原图像以cv2::BGR格式返回。

        :return: cv2::BGR格式的原图。
        """
        return self.srcImage.astype(np.uint8)
