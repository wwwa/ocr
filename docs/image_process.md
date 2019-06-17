# 图像处理

### 读取图像
cv2.imread(img,flag)

通常图像每个像素点的颜色我们以RGB的格式来描述(或者RGBA),可以通过三基色(red,green,blue)来描述所有颜色,对于透明图片我们会增加一个a(alpha)来描述其颜色的透明度.
```markdown
    cv2.IMREAD_COLOR : 读入图片,任何与透明度相关通道的会被忽视,默认以这种方式读入.
    cv2.IMREAD_GRAYSCALE : 以灰度图的形式读入图片.
    cv2.IMREAD_UNCHANGED : 保留读取图片原有的颜色通道.
```
可以简单的用-1,0,1来分别表示这3个flag

### 阈值处理

#### cv2.threshold(src, thresh, maxval, type, dst=None)
当像素值高于阀值时，我们给这个像素赋予一个新值（可能是白色），否则我们给它赋予另外一种颜色（也许是黑色）。

- src:原始图像
- thresh: 用来对像素值进行分类的阀值, 像素值上限
- maxval: 当像素值高于（或者小于）阀值时，应该被赋予新的像素值
- type: 方法选择参数，常用的有： 
    - cv2.THRESH_BINARY（黑白二值） 
    - cv2.THRESH_BINARY_INV（黑白二值反转） 
    - cv2.THRESH_TRUNC （得到的图像为多像素值） 
    - cv2.THRESH_TOZERO 
    - cv2.THRESH_TOZERO_INV 

这个函数有两个返回值，第一个为retVal，后面会解释，第二个就是阀值化之后的结果图像了。

```python
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    
    img = cv2.imread('719100.jpg',0)
    ret , thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret , thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret , thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
    ret , thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
    ret , thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
    
    titles = ['original image','Binary','binary-inv','trunc','tozero','tozero-inv']
    images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]
    
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    
    plt.show()
```

#### adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C, dst=None)
根据图像上的每一个小区域计算与其对应的阀值。因此在同一幅图像上的不同区域采用的是不同的阀值，从而使我们能在亮度不同的情况下得到更好的结果。

- src:原始图像
- thresh: 用来对像素值进行分类的阀值, 像素值上限
- adaptiveMethod: 自适应方法Adaptive Method:
    - cv2.ADAPTIVE_THRESH_MEAN_C ：领域内均值 
    - cv2.ADAPTIVE_THRESH_GAUSSIAN_C ：领域内像素点加权和，权 重为一个高斯窗口
- thresholdType：只有cv2.THRESH_BINARY 和cv2.THRESH_BINARY_INV
- blockSize: 规定领域大小（一个正方形的领域）， 一般是3，5，7，9等等
- C: 这就是一个常数，阀值就等于的平均值或者加权平均值减去这个常数

```python
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    
    img = cv2.imread('719100.jpg',0)
    #中值滤波
    img = cv2.medianBlur(img,5)
    
    ret , th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # 11为block size，2为C值
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C , cv2.THRESH_BINARY,11,2 )
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY,11,2)
    
    titles = ['original image' , 'global thresholding (v=127)','Adaptive mean thresholding',
              'adaptive gaussian thresholding']
    images = [img,th1,th2,th3]
    
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    
    plt.show()
```
### Otsu's二值化
前面对于阈值的处理上，我们选择的阈值都是127，那么实际情况下，怎么去选择这个127呢？有的图像可能阈值不是127得到的效果更好。那么这里我们需要算法自己去寻找到一个阈值，而Otsu’s就可以自己找到一个认为最好的阈值。并且Otsu’s非常适合于图像灰度直方图具有双峰的情况，他会在双峰之间找到一个值作为阈值，对于非双峰图像，可能并不是很好用。那么经过Otsu’s得到的那个阈值就是函数cv2.threshold的第一个参数了。因为Otsu’s方法会产生一个阈值，那么函数cv2.threshold的的第二个参数（设置阈值）就是0了，并且在cv2.threshold的方法参数中还得加上语句cv2.THRESH_OTSU。

```python
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    img = cv2.imread('719100.jpg',0)
    
    ret1,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    
    ret2,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #(5,5)为高斯核的大小，0为标准差
    blur= cv2.GaussianBlur(img,(5,5),0)
    
    #阀值一定要设为0
    ret3,th3=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    images=[img,0,th1,
             img,0,th2,
             img,0,th3]
    titles =['original noisy image','histogram','global thresholding(v=127)',
              'original noisy image','histogram',"otsu's thresholding",
              'gaussian giltered image','histogram',"otus's thresholding"]
    #这里使用了pyplot中画直方图的方法，plt.hist要注意的是他的参数是一维数组
    #所以这里使用了（numpy）ravel方法，将多维数组转换成一维，也可以使用flatten方法
    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])
        
    plt.show()
```
