# 四坐标目标检测数据增广
-------------------------
## 数据格式
* TXT
八个坐标点，如下:
```````text
158,131,598,131,598,177,158,177
192,194,566,194,566,237,192,237
170,346,631,346,631,380,170,380
134,386,628,386,628,427,134,427
136,427,314,427,314,466,136,466
363,693,460,693,460,736,363,736
567,695,632,695,632,732,567,732

```````
## 方法
* 代码中使用了批量数据处理，每张图片生成10张
* 增广的方法如下，由blur，rotate，guass，椒盐噪声等
````
r = np.random.randint(-3, 3)
resize = random.uniform(0.9, 1)
dw = np.random.randint(-50, 50)
dh = np.random.randint(-50, 50)
img, boxs = rotate(os.path.join(path, name), os.path.join(path, name[:-4] + '.txt'),
                   r=r,
                   resize=resize, show=False,
                   dw=dw, dh=dh)
if random.random() > 0.8:
    img = gasuss_noise(img, random.uniform(0, 0.1), random.uniform(0, 0.1))
if random.random() > 0.8:
    img = sp_noise(img, random.uniform(0, 0.1), )
if random.random() > 0.8:
    num = np.random.randint(3, 5)
    img = cv2.blur(img, (num, num))
````