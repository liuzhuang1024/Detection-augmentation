import cv2
import numpy as np
import os
import glob
import tqdm


def load_txt(path):
    return np.loadtxt(path, dtype=float, delimiter=',')


def load_img_shape(img_path):
    shape = cv2.imread(img_path).shape
    shape = (shape[1], shape[0])
    return shape


def main():
    for _ in tqdm.tqdm(glob.glob('data_set/*.jpg')):
        print(_)
        all_location = load_txt(_.replace('.jpg', '.txt'))
        shape = load_img_shape(_)
        all_location = all_location.reshape(-1, 2)
        # print(all_location)
        # for corr in all_location:
        #     if corr[0] > shape[0] or corr[1] > shape[1] or corr[0] < 0 or corr[1] < 0:
        #         print(corr, shape)
        all_location[:, 0][all_location[:, 0] > shape[0]] = shape[0]
        all_location[:, 1][all_location[:, 1] > shape[1]] = shape[1]
        all_location[:, 0][all_location[:, 0] < 0] = 0
        all_location[:, 1][all_location[:, 1] < 0] = 0
        for corr in all_location:
            if corr[0] > shape[0] or corr[1] > shape[1] or corr[0] < 0 or corr[1] < 0:
                print(corr, shape)
        # print(all_location.reshape(-1, 8).astype(str).tolist())
        with open(_.replace('.jpg', '.txt'), 'w') as f:
            f.writelines(all_location.reshape(-1, 8).astype(str).tolist())
