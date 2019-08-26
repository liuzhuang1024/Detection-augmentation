import cv2
import numpy as np
import random
import test_location


def rotate(img_name, txt_name, r=np.random.randint(-10, 10),
           resize=random.uniform(0.8, 1), show=False, save=False, save_name='1.jpg',
           dw=np.random.randint(-100, 100), dh=np.random.randint(-100, 100)):
    """

    :param img_name:
    :param txt_name:
    :param r:
    :param resize:
    :param show: bool
    :param dw:
    :param dh:
    :return:
    """

    def Srotate(angle, valuex, valuey, pointx=0, pointy=0, resize=resize, ):
        sRotatex = (valuex - pointx) * np.cos(angle) + (valuey - pointy) * np.sin(angle) + pointx
        sRotatey = (valuey - pointy) * np.cos(angle) - (valuex - pointx) * np.sin(angle) + pointy
        return int(sRotatex * resize), int(sRotatey * resize)

    img = cv2.imread(img_name, )
    copy = img.copy()
    bboxs = np.loadtxt(txt_name, dtype=int, delimiter=',')
    if show:
        for i, box in enumerate(bboxs):
            box = box.reshape((-1, 4, 2))
            cv2.fillConvexPoly(copy, box, (255, 255, 0), lineType=1)
        cv2.namedWindow('src', 1500)
        cv2.imshow('src', copy)
    imgInfo = img.shape
    height = imgInfo[0]
    width = imgInfo[1]
    deep = imgInfo[2]

    # 定义一个旋转矩阵
    matRotate = cv2.getRotationMatrix2D((0, 0), r, resize)  # mat rotate 1 center 2 angle 3 缩放系数
    dst = cv2.warpAffine(img, matRotate, (int(width * resize), int(height * resize)), borderValue=(250, 250, 250))
    M = np.float32([[1, 0, dw], [0, 1, dh]])
    dst = cv2.warpAffine(dst, M, (int(width * resize), int(height * resize)), borderValue=(250, 250, 250))
    new_img = np.copy(dst)
    new_boxs = []
    for box in bboxs:
        tmp = []
        for point in box.reshape((4, 2)):
            p = Srotate(r * 3.14 / 180, valuex=point[0], valuey=point[1])
            tmp.append([p[0] + dw, p[1] + dh])
        # wh = (tmp[2][0] - tmp[0][0]) / (tmp[2][1] - tmp[0][1])
        # if wh > 1000 or wh < 0.0001:
        #     continue
        new_boxs.append(tmp)
    if save or show:
        for box in np.array(new_boxs).reshape(-1, 8):
            # box = np.array(tmp, dtype=int).reshape((4, 2))
            # cv2.fillConvexPoly(dst, box, (255, 255, 0), lineType=1)
            cv2.rectangle(dst, (box[0], box[1]), (box[4], box[5]), (0, 255, 0), 6)
    if save:
        cv2.imwrite(save_name, dst)
    if show:
        cv2.namedWindow('dst', 1500)
        cv2.imshow('dst', dst)
        cv2.waitKey(0)
    return new_img, new_boxs


if __name__ == '__main__':
    # /home/lz/下载/fayuan
    # /home/lz/立案识别模板/立案识别模板/templates
    # /home/lz/下载/all
    import os
    import tqdm
    from noise import sp_noise, gasuss_noise

    path = '/home/lz/yijing'
    output = 'data_set'
    if not os.path.exists(output):
        os.makedirs(output)
    for name in tqdm.tqdm(os.listdir(path)):
        if '.jpg' in name or '.png' in name:
            if os.path.exists(os.path.join(path, name[:-4] + '.txt')):
                for _ in range(10):
                    r = np.random.randint(-1, 1)
                    resize = random.uniform(0.999, 1)
                    dw = np.random.randint(-300, 300)
                    dh = np.random.randint(-300, 300)
                    img, boxs = rotate(os.path.join(path, name), os.path.join(path, name[:-4] + '.txt'),
                                       r=r, save_name=os.path.join('save', name),
                                       resize=resize,
                                       dw=dw, dh=dh)
                    if random.random() > 0.8:
                        img = gasuss_noise(img, random.uniform(0, 0.05), random.uniform(0, 0.05))
                    if random.random() > 0.8:
                        img = sp_noise(img, random.uniform(0, 0.05), )
                    if random.random() > 0.8:
                        num = np.random.randint(3, 5)
                        img = cv2.blur(img, (num, num))
                    # if random.random() > 0.8:
                    #     contrast = random.randint(50, 100)  # 对比度
                    #     brightness = random.randint(50, 100)  # 亮度
                    #     img = cv2.addWeighted(img, contrast, img, 0, brightness)
                    cv2.imwrite(os.path.join(output, name[:-4] + '_noise_blur_' + str(_).zfill(5) + '.jpg'), img)
                    # 判断异常的bbox
                    boxs = np.array(boxs).reshape(-1, 2)
                    shape = [img.shape[1], img.shape[0]]
                    boxs[:, 0][boxs[:, 0] > shape[0]] = shape[0]
                    boxs[:, 1][boxs[:, 1] > shape[1]] = shape[1]
                    boxs[:, 0][boxs[:, 0] < 0] = 0
                    boxs[:, 1][boxs[:, 1] < 0] = 0
                    for corr in boxs:
                        if corr[0] > shape[0] or corr[1] > shape[1] or corr[0] < 0 or corr[1] < 0:
                            print(corr, shape)
                    # 保存bbox
                    boxs = np.array(boxs, dtype=str).reshape(-1, 8)
                    for box in boxs.astype(int):
                        # box = np.array(tmp, dtype=int).reshape((4, 2))
                        # cv2.fillConvexPoly(dst, box, (255, 255, 0), lineType=1)
                        cv2.rectangle(img, (box[0], box[1]), (box[4], box[5]), (0, 255, 0), 6)
                    cv2.imwrite(os.path.join('save', 'noise_blur' + str(_).zfill(5) + name), img)
                    with open(os.path.join(output, name[:-4] + '_noise_blur_' + str(_).zfill(5) + '.txt'), 'w') as f:
                        for box in boxs:
                            f.write(','.join(list(box)) + '\n')
