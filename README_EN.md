**Seam Carving Algorithm's Python Efficient Implementation**

[![Author: Razirp](https://img.shields.io/badge/author-Razirp-cc0000)](https://github.com/Razirp) [![License](https://img.shields.io/github/license/Razirp/Seam-Carving-python)](https://github.com/Razirp/Seam-Carving-python/blob/main/LICENSE) [![Language: Python](https://img.shields.io/badge/Language-Python-blue)](https://cppreference.com/) [![Built with OpenCV](https://img.shields.io/badge/Built%20with-OpenCV-%232737A6?style=flat-square&logo=opencv&logoColor=white&link=https://opencv.org)](https://opencv.org) [![CSDN Blog](https://img.shields.io/static/v1?label=CSDN%20Blog&message=Visit&color=e96140&style=flat-square)](https://blog.csdn.net/qq_45899276/article/details/138096548?csdn_share_tail=%7B%22type%22%3A%22blog%22%2C%22rType%22%3A%22article%22%2C%22rId%22%3A%22138096548%22%2C%22source%22%3A%22qq_45899276%22%7D) ![HitCount](https://img.shields.io/endpoint?url=https%3A%2F%2Fhits.dwyl.com%2FRazirp%2FSeam-Carving-python.json%3Fcolor%3Dff9900) [![GitHub stars](https://img.shields.io/github/stars/Razirp/Seam-Carving-python)](https://github.com/Razirp/Seam-Carving-python/stargazers)

📖 中文版点[这里](README.md)！

> A high-performance Python implementation of the Seam Carving algorithm.
>
> The code in this repository is based on the implementation at https://github.com/vivianhylee/seam-carving, but it has been optimized for performance, with the algorithm's execution speed reaching 20 to 40 times faster than the code in the aforementioned repository!
> - For specific optimization analysis, see Section 2.4.

**I. Experimental Principle**

Scaling is a common image processing operation. Adjusting only geometric parameters is not the most ideal approach, as the importance of each pixel in an image is closely related to the image content.

The experiment requires designing an algorithm that takes an original image as input and outputs an image with a changed aspect ratio (the aspect ratio can be specified at will). The algorithm must consider differential processing of pixels.

The Seam Carving algorithm used in this experiment [1] can achieve content-aware image resizing - changing the size of the image while maintaining the important features of the image.

When reducing the size of the image, we prefer to remove more low-energy pixels and retain more high-energy pixels.

When enlarging the image size, we appropriately choose the order of seam insertion to ensure a balance between the original image content and the artificially inserted pixels.

**Basic Concepts**

**Energy Function**

The importance of pixels is defined using an energy function.

There are several types of energy functions to choose from:

- Gradient Magnitude
- Entropy
- Visual Salience
- Eye Movement

A simple gradient-based energy function is:

\[ e_1(I) = | \frac{\partial}{\partial x} I | + | \frac{\partial}{\partial y} I | \]

where \( I \) is an \( n \times m \) image.

**Seam**

A Seam refers to a continuous path of low-energy pixels that traverse the image from top to bottom or left to right.

We can change the size of the image in the horizontal or vertical direction by inserting or removing seams.

For an \( n \times m \) image, a vertical seam is defined as:

\[ s^X = \{s_i^x \}_{i=1}^n = \{(x(i), i)\}_{i=1}^n, \text{ s.t. } \forall i, |x(i) - x(i-1)| \le 1 \]

where \( x \) is a mapping from \( [1, ..., n] \) to \( [1, ..., m] \). In other words, a vertical seam is an 8-connected path of pixels from top to bottom in the image, with each row containing only one pixel.

Similarly, a horizontal seam can be defined as:

\[ s^Y = \{s_j^y\}_{j=1}^m = \{(j, y(j))\}_{j=1}^m, \text{ s.t. } \forall j, |y(j) - y(j-1)| \le 1 \]

The constraints in the above two definitions can be changed from \( \le 1 \) to \( \le k \) as needed.

**Seam Cost**

Given an energy function \( e \), the seam cost can be defined as:

\[ E(s) = E(I_s) = \sum_{i=1}^n e(I(s_i)) \]

**Optimal Seam**

We look for the optimal seam \( s \) that minimizes the seam cost:

\[ s = \min_s E(s) = \min_s \sum_{i=1}^n e(I(s_i)) \]

Dynamic programming can be used to find the optimal seam:

For vertical seams:

\[ M(i, j) = e(i, j) + \min(M(i-1, j-1), M(i-1, j), M(i-1, j+1)) \]

The minimum value in the last row of \( M \) indicates the end of the minimum vertical seam.

Horizontal seams are similar, with:

\[ M(i, j) = e(i, j) + \min(M(i-1, j-1), M(i, j-1), M(i+1, j-1)) \]

**Algorithm Steps**

The algorithm has no parameters during the process of removing or inserting seams.

Users can add weights to the image's energy and guide the desired results.

Let the original size of the image be \( n \times m \), and the specified size of the scaled image be \( n' \times m' \). Let \( r = (m - m') \), \( c = (n - n') \), \( k = r + c \).

**Single Direction Change**

**Reduction**

When \( r > 0, c = 0 \), the horizontal size of the image remains unchanged, and the vertical size is reduced, requiring the removal of \( r \) horizontal seams. The basic steps of the algorithm at this time are as follows:

1. Calculate the energy of each pixel based on the selected energy function to obtain the energy map of the image.
2. Use the dynamic programming method mentioned above to find the seam with the minimum energy in the horizontal direction.
3. Remove all pixels corresponding to the Seam from the image to obtain a new image.
4. Repeat the above steps \( r \) times to obtain the scaled image.

**Expansion**

For the reduction of image size, we only need to find the corresponding seams and remove them in sequence; however, when \( r < 0 \) or \( c < 0 \), the transformed size will be larger than the original size, which requires us to insert artificially generated pixels. Following the basic steps of the algorithm, we naturally think of copying a seam after finding the energy minimum seam and inserting it into the image. But in fact, this operation will not change the chosen energy minimum seam, that is, we will continue to copy the same seam to produce a stretched image, which is not the ideal result.

To address this issue caused by image expansion, reference [1] proposes a solution. Taking the case of \( r < 0 \) as an example, we need to insert \( -r \) horizontal seams at this time, so we can find the energy of the top \( -r \) smallest \( -r \) seams at one time, then copy them and insert them next to the corresponding seams, so that we can expand the size of the image by \( -r \) units at one time to achieve the goal of image adjustment. At the same time, in order to avoid destroying the important content in the image when the expansion (such as more than 50%) is too large, we can also divide the expansion process into several steps, and the expansion of each step will not exceed a part of the previous step to ensure that the important content of the image is not stretched.

That is, the basic steps of the algorithm for expansion (taking \( r < 0, c = 0 \) as an example, at this time, \( |r| = -r \) seams need to be inserted) are as follows:

1. Set the remaining number of seams to be inserted as \( r' \). The initial value \( r' = -r \).
2. Calculate the energy of each pixel based on the selected energy function to obtain the energy map of the image.
3. Use the dynamic programming method mentioned above to find the \( t = \min(r', p) \) smallest seams in the horizontal direction.
   - Here \( p \) can be set to the product of the size \( m \) in that direction and a certain proportionality constant, such as \( p = 0.5m \).
4. Copy the found seams and insert them into the image to obtain a new image, \( r' = r' - t \).
5. Repeat steps 2 to 4 until \( r' == 0 \). Obtain the scaled image.

**Two-Direction Change**

The above discussion only addresses changes in one dimension, that is, a single horizontal or vertical change. When the change extends to two dimensions, we must consider the optimal order of seam operations: that is, whether to choose a horizontal seam or a vertical seam for operation in each step.

**Relocation Based on Optimal Seam Order**

The optimal order satisfies the following function:

\[ \min_{s^x, s^y, \alpha} \sum_{i=1}^{k} E(\alpha_i s_i^x + (1-\alpha_i) s_i^y) \]

where \( \alpha_i \) indicates whether to remove a vertical seam or a horizontal seam in the \( i \)-th step. \( k = r + c \), \( r = (m - m') \), \( c = (n - n') \).

Dynamic programming can be used to solve this:

\[ T(r, c) = \min(T(r-1, c) + E(s^x(I_{(n-r-1) \times (m-c)})), T(r, c-1) + E(s^y(I_{(n-r) \times (m-c-1)}))) \]

\( T(r, c) \) indicates the minimum cost required to obtain an image of size \( (n-r) \times (m-c) \). Among them, \( T(0,0) = 0 \).

Record the choices made at each step of dynamic programming, and backtracking from \( T(r, c) \) to \( T(0,0) \) can determine the optimal order of seam operations.

**II. Experimental Steps**

2.1 Understand the Experimental Principle

Learn about common image scaling techniques and the advantages and basic principles of the Seam Carving method by studying literature [1] and referring to several online blogs.

2.2 Configure the Experimental Environment

The experimental environment for this experiment is as follows:

- Windows 11 Professional Edition 21H1
- Python 3.8.5
- OpenCV-Python 4.5.5.62
- NumPy 1.20.1
- tqdm 4.62.3
- PyCharm 2021.2.2 Community Edition

2.3 Algorithm Process Design

The first part of this report has briefly introduced the basic ideas of the algorithm used in the experiment. This part will mainly introduce the basic process of the algorithm implemented through Python code, without touching on technical details. The core concepts of the algorithm, such as energy map processing, and the improvements made in the experiment based on previous work, will be introduced in detail in section 2.4.

To facilitate the use and maintenance of the program, the implementation of the algorithm in this experiment is packaged in a class called SeamCarver. The following methods are all methods of the class SeamCarver.

Initialization

The `__init__` method is the entry part of the solution, which initializes the object based on the input parameters and calls the algorithm module to start the image cropping:

```python
def __init__(self, inputPath, targetHeight, targetWidth):
    """
    Initialize the fields and obtain the target image.
    """
```

Decide the Cropping Scheme Based on Size

The last call in the `__init__` method is the `start()` method. `start()` will decide the cropping scheme to be used based on the input target size:

```python
def start(self):
    """
    The beginning part of the Seam Carving algorithm.
    Choose different operational steps based on the change in image size requirements:
    When only one direction's size changes, select one-dimensional operation (column direction or row direction); otherwise, select two-dimensional operation.
    - Column direction refers to the seams being vertical
    - Row direction refers to the seams being horizontal
    """
```

Three Cropping Schemes

`colsSeamCarve()`

This is the main cropping scheme, and the other two cropping schemes are based on the scheme for cropping vertical seams.

```python
def colsSeamCarve(self, delta_cols):
    """
    Only the number of columns changes. Adds (or removes if negative) delta_cols column seams to tempImage.
    """
```

`rowsSeamCarve()`

Transform the problem into handling vertical seams by transposing, and then transform back to the original problem domain to get the expected result.

```python
def rowsSeamCarve(self, delta_rows):
    """
    Only the number of rows changes. Adds (or removes if negative) delta_rows row seams to tempImage.
    """
```

`twoDimSeamCarve()`

Call the above two methods, first process column changes and then row changes.

```python
def twoDimSeamCarve(self, delta_rows, delta_cols):
    """
    When changes occur in both directions, first process column changes and then row changes.
    """
```

Handling of Vertical Seams

It can be seen from the previous part that handling vertical seams is the main part of the algorithm and the basis for all cases.

Depending on the expansion or reduction of the size, we may insert or delete vertical seams.

Removing Seams

Each loop performs three steps: checking the energy map, finding the seam with the minimum energy, and removing the seam. Each loop can remove one seam.

```python
def removeColSeams(self, num_seamsRemoved):
    """
    Remove num_seamsRemoved column (vertical) seams from tempImage.
    """
```

Inserting Seams

The suggestion in reference [1] is to insert in multiple steps, finding the top \( k \) seams with the lowest energy for insertion at a time.

The program will judge the number of seams that need to be inserted. If it exceeds a certain proportion of the corresponding size, it will choose to insert in multiple times.

In each individual insertion, the position of the seam will be located first, and then the insertion will be made:

First, delete \( k \) seams in turn in the temporary image, and record the positions of the deleted seams as the selected \( k \) seams.
Since each selected seam will be deleted from the original image, it can be ensured that the \( k \) seams found do not overlap in pixels.
Since each selected seam is the seam with the lowest energy remaining in the image, it can be ensured that the \( k \) seams found are the \( k \) seams with the lowest energy in the original image.
Each time a seam is deleted, the structure of the image changes, which may cause a deviation in the recording of the seams, and it is necessary to dynamically correct when inserting.
Then, insert the seam into the original image according to the recorded positions in the previous step.
Due to the third point mentioned above, the order of seam insertion must be consistent with the order in which the seams were deleted in the previous step.
Due to the third point mentioned above, after inserting each seam, all remaining seam records need to be dynamically updated.

```python
def insertColSeams(self, num_seamsInserted):
    """
    Determine whether multiple insertions are needed:
    - When the expansion ratio exceeds \( p \) times the original size, it will be inserted in multiple times, with the scale of each insertion being 0.8 times the previous one, until the requirement is met.
    def insertColSeams_sub(self, num_seamsInserted):
    """
    Insert num_seamsInserted column (vertical) seams into tempImage.
    """
```

2.4 Core Technical Details and Improvements on Previous Work

In the implementation of core technical details, I have referred to materials including reference [1] and the code provided by the assignment (https://github.com/vivianhylee/seam-carving), and have made improvements in several parts. Some improvements are at the algorithm level, which have reduced the time complexity of the algorithm to a certain extent; some improvements are made in view of the characteristics of the Python language, which are improvements at the engineering implementation level:

Improved the calculation and processing method of the energy map, reducing the time complexity of energy map processing.

Improved the code implementation scheme of dynamic programming, greatly improving the program's running speed.

Fully utilized the advantages of Python and NumPy in handling multi-dimensional arrays, simplifying the way of writing code, reducing the amount of code, making the code more concise and improving the running efficiency.

Since the main part of the algorithm is written for vertical seams (see 2.3 - Three Cropping Schemes), all the "seams" mentioned below refer to vertical seams.

Energy Map

The calculation of the image energy map is an important part of the Seam Carving algorithm, especially the seam removal part. Reference [1] provides several different energy functions, mainly based on the sum of the absolute values of the two-directional gradients.

The code uses the following scheme, utilizing:

OpenCV's official documentation states that:

cv.split() is a costly operation (in terms of time). So only do it when you need to, otherwise use NumPy indexing.

```python
self.energyMap = np.absolute(cv2.Scharr(self.tempImage, -1, 1, 0)).sum(2) + \
                 np.absolute(cv2.Scharr(self.tempImage, -1, 0, 1)).sum(2)
```

At the same time, the scheme adopted in reference [1] and the code given in the assignment is to recalculate and generate a new energy map for the entire image every time a seam is removed. However, in fact, the cv2.Scharr operator uses a \( 3 \times 3 \) template, so removing a seam will affect the energy values of at most 1 pixel to the left and right, and the energy of most other pixels in the image will not be affected. Therefore, there is a lot of room for optimization in the algorithm on this point: if only the surrounding part of the energy map of the seam is updated each time, the time complexity of the energy map update can be reduced from \( O(mn) \) to \( O(m) \).

The author considered trading space for time and allocated a field self.energyMap specifically to store the energy map, so only the corresponding update of the energy map is needed after each seam is deleted.

The author first tried the method of locating the positions of the pixels that need to be updated based on the seam removed each time, and then recalculating the energy of each pixel one by one. However, this method requires a large number of iterative loops. Considering the low efficiency of Python's iteration (which will be particularly obvious in the next section), this is not a good optimization solution. Therefore, the author chose the second best, using the method of re-filtering and recalculating a part of the image with the cv2.Scharr operator. Because the underlying code of cv2.Scharr() is written in C/C++, its iterative operation efficiency will be significantly higher than that of Python.

The algorithm will re-filter the image area between one unit outward from the left end of the seam to one unit outward from the right end of the seam each time it is updated. Considering that vertical seams generally do not span a large area horizontally, this can still significantly reduce the time complexity of the algorithm and make good use of the advantages of high iteration efficiency of C/C++ language:

```python
def updateEnergyMap(self, removedSeam):
    """
    Update the energy map based on the image change after each seam is removed.
    - Re-filter the image area between the left and right boundaries (each expanded by one grid) of the seam using the cv2.Scharr operator.
    """
```

In this way, there is no need to recalculate the energy map once before each execution of the seam deletion operation, but just call checkEnergyMap to check whether the energy map is available:

```python
def checkEnergyMap(self):
    """
    Check if the energy map is available, and if not, recalculate the energy map.
    :return: No return value.
    """
    if self.energyMap is None:  # If the gradient map has not been calculated, perform the initial calculation
        self.energyMap = np.absolute(cv2.Scharr(self.tempImage, -1, 1, 0)).sum(2) + \
                         np.absolute(cv2.Scharr(self.tempImage, -1, 0, 1)).sum(2)
```

Finding the Seam with the Minimum Total Energy

In the seam removal, the core part is the algorithm for finding the seam with the minimum energy. The paper [1] proposes a dynamic programming approach to solve this problem. The implementation given in the assignment's code provides a realization of dynamic programming:

```python
# This is the implementation in the given code of the assignment, for reference only, not the scheme adopted in this experiment.
def cumulative_map_backward(self, energy_map):
    m, n = energy_map.shape
    output = np.copy(energy_map)
    for row in range(1, m):
        for col in range(n):
            output[row, col] = \
                energy_map[row, col] + np.amin(output[row - 1, max(col - 1, 0): min(col + 2, n - 1)])
    return output
```

On the algorithm level, this implementation is correct and fully conforms to the description in paper [1]. However, it can be seen that there are double nested loops in the code. Considering the low efficiency of Python's iteration, such writing may significantly increase the program's running time. In addition, the author learned from communication with classmates that the running speed of the code given in the assignment is much slower than that of the program implemented in C++ language; and the author tested that the most time-consuming part of the algorithm is this part of dynamic programming: it takes about 1 second on average to delete a seam, and the dynamic programming step can take up to \( 0.8 \sim 0.9 \) seconds. Therefore, the author decided to start with this double loop iteration and try to optimize the program's running speed.

This problem is caused by the low efficiency of Python's iteration, so the natural idea is to try to transfer more computational work to the C/C++ environment for execution. Considering that the underlying implementation of the NumPy library is in C/C++, and the results of each row in dynamic programming are only related to the previous row, the author tried to use the NumPy library to process each row of the dynamic programming matrix as a whole to reduce a layer of iterative loops:

```python
def getDPMap_colSeams(self):
    """
    Get the dynamic programming map for calculating vertical seams.
    """
```

After testing, the running speed of the optimized code was increased by \( 20 \sim 40 \) times compared to the original code, greatly improving the running speed of the algorithm.

By backtracking on the obtained dynamic programming map, the index list of the seam with the minimum total energy can be determined:

```python
def findOptimalSeam_col(self, dpMap):
    """
    Backtrack from dpMap to obtain the index list of the seam with the minimum total energy.
    """
```

Removing or Inserting Seams in the Image

The algorithm operations for removing and inserting seams are similar, both involve updating each row of the image and performing the corresponding deletion or insertion operation at the position of the seam.

Comparing the code, this experiment fully utilizes the advantages of Python and NumPy in handling multi-dimensional arrays, simplifying the way of writing code, reducing the amount of code, making the code more concise and improving the running efficiency:

```python
def doColSeamRemoval(self, seamToRemove):
    """
    Remove the vertical seam indicated by seamsToRemove from tempImage.
```

```python
def doColSeamInsertion(self, seamToInsert):
    """
    Insert a new seam at the position of seamToInsert, with pixel values taken as the average of the left and right pixel values.
    """
```

According to the needs of the algorithm, it is necessary to call updateEnergyMap() to update the energy map after removing the seam. After inserting the seam, it is necessary to update the index of the remaining seams that need to be inserted:

See 2.3 - Handling of Vertical Seams - Inserting Seams for details.

```python
def updateSeamsQueue(self, seamsQueue, lastInsertedSeam):
    """
    Update the coordinates in seamsQueue based on the most recently inserted seam and return.
    """
```

2.5 Program Run and Algorithm Testing

main.py provides an example of running the program to test the algorithm:

```python
import os.path
import cv2
from SeamCarving import SeamCarver

if __name__ == '__main__':

```

**III. Experimental Results and Analysis**

Input

The input image is as follows, with the original size being: height × width = 384 × 512.

Output

1. Single Direction Size Change

The following result images have only one direction of size change, and the size of the other direction remains unchanged.

Columns reduced, rows unchanged - height × width: 384 × 512 → 384 × 300

Columns increased, rows unchanged - height × width: 384 × 512 → 384 × 800

Rows reduced, columns unchanged - height × width: 384 × 512 → 200 × 512

Rows increased, columns unchanged - height × width: 384 × 512 → 500 × 512

2. Multi-Direction Size Change

The following result images have changes in both directions of size.

Rows increased, columns increased: height × width: 384 × 512 → 400 × 800

Rows increased, columns reduced: height × width: 384 × 512 → 400 × 400

Rows reduced, columns increased: height × width: 384 × 512 → 300 × 700

Rows reduced, columns reduced: height × width: 384 × 512 → 250 × 350

3. Program Print Output

This experiment has added visual output in the program to show the user the progress of the algorithm in real-time. The program's print output during the execution of the above operations is as follows:

```
已读入图像in\image.jpg
算法操作开始...执行变化：
高×宽： 384 × 512 ==> 384 × 300
正在执行removeSeams()...将移除 212 条接缝：
100%|██████████| 212/212 [00:07<00:00, 26.57it/s]
处理完毕，处理时间共：7.9979秒
-----------------------------------------------------------
图像已经存储到： out\image_0_re.jpg
-----------------------------------------------------------
算法操作开始...执行变化：
高×宽： 384 × 512 ==> 384 × 800
正在执行insertSeams()...将插入 288 条接缝：
正在定位接缝位置：
100%|██████████| 288/288 [00:07<00:00, 38.64it/s]
正在插入接缝：
100%|██████████| 288/288 [00:05<00:00, 49.82it/s]
处理完毕，处理时间共：13.2346秒
-----------------------------------------------------------
图像已经存储到： out\image_0_en.jpg
-----------------------------------------------------------
算法操作开始...执行变化：
高×宽： 384 × 512 ==> 200 × 512
正在执行removeSeams()...将移除 184 条接缝：
100%|██████████| 184/184 [00:05<00:00, 34.54it/s]
处理完毕，处理时间共：5.3268秒
-----------------------------------------------------------
图像已经存储到： out\image_re_0.jpg
-----------------------------------------------------------
算法操作开始...执行变化：
高×宽： 384 × 512 ==> 500 × 512
正在执行insertSeams()...将插入 116 条接缝：
正在定位接缝位置：
100%|██████████| 116/116 [00:03<00:00, 33.43it/s]
正在插入接缝：
100%|██████████| 116/116 [00:02<00:00, 42.55it/s]
处理完毕，处理时间共：6.1964秒
-----------------------------------------------------------
图像已经存储到： out\image_en_0.jpg
-----------------------------------------------------------
算法操作开始...执行变化：
高×宽： 384 × 512 ==> 400 × 800
正在执行insertSeams()...将插入 288 条接缝：
正在定位接缝位置：
100%|██████████| 288/288 [00:07<00:00, 40.50it/s]
正在插入接缝：
100%|██████████| 288/288 [00:05<00:00, 50.92it/s]
正在执行insertSeams()...将插入 16 条接缝：
正在定位接缝位置：
100%|██████████| 16/16 [00:00<00:00, 20.65it/s]
正在插入接缝：
100%|██████████| 16/16 [00:00<00:00, 27.64it/s]
处理完毕，处理时间共：14.1402秒
-----------------------------------------------------------
图像已经存储到： out\image_en_en.jpg
-----------------------------------------------------------
算法操作开始...执行变化：
高×宽： 384 × 512 ==> 400 × 400
正在执行removeSeams()...将移除 112 条接缝：
100%|██████████| 112/112 [00:02<00:00, 39.02it/s]
正在执行insertSeams()...将插入 16 条接缝：
正在定位接缝位置：
100%|██████████| 16/16 [00:00<00:00, 40.89it/s]
正在插入接缝：
100%|██████████| 16/16 [00:00<00:00, 56.89it/s]
处理完毕，处理时间共：3.5426秒
-----------------------------------------------------------
图像已经存储到： out\image_en_re.jpg
-----------------------------------------------------------
算法操作开始...执行变化：
高×宽： 384 × 512 ==> 300 × 700
正在执行insertSeams()...将插入 188 条接缝：
正在定位接缝位置：
100%|██████████| 188/188 [00:04<00:00, 38.94it/s]
正在插入接缝：
100%|██████████| 188/188 [00:03<00:00, 52.39it/s]
正在执行removeSeams()...将移除 84 条接缝：
100%|██████████| 84/84 [00:03<00:00, 24.76it/s]
处理完毕，处理时间共：11.8186秒
-----------------------------------------------------------
图像已经存储到： out\image_re_en.jpg
-----------------------------------------------------------
算法操作开始...执行变化：
高×宽： 384 × 512 ==> 250 × 350
正在执行removeSeams()...将移除 162 条接缝：
100%|██████████| 162/162 [00:04<00:00, 39.68it/s]
正在执行removeSeams()...将移除 134 条接缝：
100%|██████████| 134/134 [00:02<00:00, 51.96it/s]
处理完毕，处理时间共：6.6616秒
-----------------------------------------------------------
图像已经存储到： out\image_re_re.jpg
-----------------------------------------------------------
```

Analysis

Based on the results, it can be seen that according to the input image and the given target aspect ratio, the algorithm can provide an output image that conforms to the corresponding aspect ratio. In addition, the important content in the images is relatively well preserved, achieving a good effect. Moreover, the experimental algorithm has a fast execution speed and has good practicality. In general, the experiment has been basically successful.

Areas for Improvement

There are still areas in the experimental algorithm that can be improved: For the case of changes in two directions at the same time, the solution given in reference [1] is to decide the direction of the seam in each step of the operation based on a dynamic programming algorithm, which is also reflected in this report (see Experimental Principle - Algorithm Steps - Two-Direction Change). However, due to time constraints and considering the low efficiency of implementing a dynamic programming algorithm with double-layer iterative loops in Python, this experiment did not implement the related algorithm, but instead used a scheme of first processing column changes and then processing row changes.

**Appendix - Code Running Method**

Configure the Experimental Environment

Ensure that the machine running the program has at least the following experimental environment:

- Python 3.x version
- OpenCV-Python library
- NumPy library
- tqdm library

Set Input and Output Paths and Target Size Parameters

Open main.py with any text editor and customize the following four fields:

```python
imageName_input = "image.jpg"  # The filename of the input image
imageName_output = "image_result.jpg"  # The filename of the output image
target_height = 400  # The target height of the scaled image
target_width = 400   # The target width of the scaled image
```

Run the Program

Make sure main.py and SeamCarving.py are in the same directory, and there are two paths ./in and ./out in the same directory. Place the input image in ./in.

Use any method you are accustomed to for executing Python files to run main.py, and you can get the output of the program in the ./out folder.

For example, in the Windows Command Prompt, type python main.py to execute the program.

**References**

Avidan, S., & Shamir, A. (2007). Seam carving for content-aware image resizing. In ACM SIGGRAPH 2007 papers (pp. 10-es).

### License
This project is licensed under the MIT License - see the `LICENSE` file for details.

### Contact
For any questions or inquiries, please contact me at razirp77@outlook.com.