# -*- coding: utf-8 -*-
"""
@Author: OpenAI Assistant
@Date: 2023-11-26

实现UCIQE和UIQM水下图像质量评价指标的Python代码
"""

import os
import cv2
import numpy as np
import time
import math
import csv
from skimage import color
from skimage.filters import sobel


class UCIQE_UIQM:
    def __init__(self, image_folders, output_csv='UCIQE_UIQM.csv', image_size=(640, 640),
                 uciqe_coeffs=(0.4680, 0.2745, 0.2576), uiqm_coeffs=(0.0282, 0.2953, 3.5753),
                 metrics_to_compute=('uciqe', 'uiqm')):
        """
        初始化UCIQE_UIQM计算类

        参数：
        - image_folders: 输入图像文件夹列表
        - output_csv: 输出CSV文件路径
        - image_size: 图像预处理大小，默认(640, 640)
        - uciqe_coeffs: UCIQE指标的系数，默认为论文中的系数
        - uiqm_coeffs: UIQM指标的系数，默认为论文中的系数
        - metrics_to_compute: 要计算的指标列表，可以是'uciqe'，'uiqm'或两者
        """
        self.image_folders = image_folders
        self.output_csv = output_csv
        self.image_size = image_size
        self.uciqe_coeffs = uciqe_coeffs
        self.uiqm_coeffs = uiqm_coeffs
        self.metrics_to_compute = metrics_to_compute

    def process_images(self):
        """
        处理图像，计算指定的指标并保存结果到CSV文件
        """
        # 打开CSV文件，准备写入
        with open(self.output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['image_name', 'image_path']
            if 'uciqe' in self.metrics_to_compute:
                fieldnames.extend(['uciqe_value', 'uciqe_time'])
            if 'uiqm' in self.metrics_to_compute:
                fieldnames.extend(['uiqm_value', 'uiqm_time'])
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            # 遍历每个文件夹
            for folder in self.image_folders:
                for root, _, files in os.walk(folder):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                            image_path = os.path.join(root, file)
                            image_name = file
                            print(f"Processing {image_name}")
                            # 读取并预处理图像
                            img = cv2.imread(image_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, self.image_size)
                            result = {'image_name': image_name, 'image_path': root}
                            # 计算UCIQE指标
                            if 'uciqe' in self.metrics_to_compute:
                                start_time = time.time()
                                uciqe_value = self.compute_uciqe(img, self.uciqe_coeffs)
                                end_time = time.time()
                                result['uciqe_value'] = uciqe_value
                                result['uciqe_time'] = end_time - start_time
                            # 计算UIQM指标
                            if 'uiqm' in self.metrics_to_compute:
                                start_time = time.time()
                                uiqm_value = self.compute_uiqm(img, self.uiqm_coeffs)
                                end_time = time.time()
                                result['uiqm_value'] = uiqm_value
                                result['uiqm_time'] = end_time - start_time
                            # 写入CSV文件
                            writer.writerow(result)

    def compute_uciqe(self, img, coeffs):
        """
        计算UCIQE指标

        参数：
        - img: 输入的RGB图像，numpy数组
        - coeffs: UCIQE指标的系数，tuple格式

        返回：
        - uciqe_value: 计算得到的UCIQE值
        """
        # 将RGB图像转换为Lab颜色空间
        img_lab = color.rgb2lab(img)
        L = img_lab[:, :, 0] / 100.0  # 亮度通道归一化到[0,1]
        a = img_lab[:, :, 1] / 128.0  # a通道归一化到[-1,1]
        b = img_lab[:, :, 2] / 128.0  # b通道归一化到[-1,1]

        # 计算色度
        chroma = np.sqrt(a ** 2 + b ** 2)
        # 色度的标准差
        sigma_c = np.std(chroma)
        # 亮度对比度，计算亮度通道的第1百分位和第99百分位
        L_flat = L.flatten()
        L_sorted = np.sort(L_flat)
        per_1 = L_sorted[int(0.01 * len(L_sorted))]
        per_99 = L_sorted[int(0.99 * len(L_sorted))]
        con_l = per_99 - per_1
        # 饱和度的均值
        saturation = chroma / (L + 1e-6)  # 避免除零
        mu_s = np.mean(saturation)

        # 计算UCIQE值
        c1, c2, c3 = coeffs
        uciqe_value = c1 * sigma_c + c2 * con_l + c3 * mu_s
        return uciqe_value

    def compute_uiqm(self, img, coeffs):
        """
        计算UIQM指标

        参数：
        - img: 输入的RGB图像，numpy数组
        - coeffs: UIQM指标的系数，tuple格式

        返回：
        - uiqm_value: 计算得到的UIQM值
        """
        c1, c2, c3 = coeffs
        uicm = self.compute_uicm(img)
        uism = self.compute_uism(img)
        uiconm = self.compute_uiconm(img)
        uiqm_value = c1 * uicm + c2 * uism + c3 * uiconm
        return uiqm_value

    def compute_uicm(self, img):
        """
        计算色彩测量指标 UICM

        参数：
        - img: 输入的RGB图像，numpy数组

        返回：
        - uicm_value: 计算得到的UICM值
        """
        R = img[:, :, 0].flatten()
        G = img[:, :, 1].flatten()
        B = img[:, :, 2].flatten()
        RG = R - G
        YB = (R + G) / 2 - B

        mu_a_RG = self.mu_a(RG)
        mu_a_YB = self.mu_a(YB)
        sigma_a_RG = self.sigma_a(RG, mu_a_RG)
        sigma_a_YB = self.sigma_a(YB, mu_a_YB)

        # 根据论文中的公式计算UICM
        l = np.sqrt(mu_a_RG ** 2 + mu_a_YB ** 2)
        r = np.sqrt(sigma_a_RG + sigma_a_YB)
        uicm_value = (-0.0268) * l + 0.1586 * r
        return uicm_value

    def mu_a(self, x, alpha_L=0.1, alpha_R=0.1):
        """
        计算非对称alpha截断均值

        参数：
        - x: 输入的一维数组
        - alpha_L: 左侧截断比例
        - alpha_R: 右侧截断比例

        返回：
        - mu: 截断均值
        """
        x_sorted = np.sort(x)
        K = len(x_sorted)
        T_a_L = int(np.ceil(alpha_L * K))
        T_a_R = int(np.floor(alpha_R * K))
        x_trimmed = x_sorted[T_a_L:K - T_a_R]
        mu = np.mean(x_trimmed)
        return mu

    def sigma_a(self, x, mu):
        """
        计算方差

        参数：
        - x: 输入的一维数组
        - mu: 均值

        返回：
        - sigma: 方差
        """
        sigma = np.mean((x - mu) ** 2)
        return sigma

    def compute_uism(self, img, window_size=10):
        """
        计算清晰度测量指标 UISM

        参数：
        - img: 输入的RGB图像，numpy数组
        - window_size: 分块大小，默认10

        返回：
        - uism_value: 计算得到的UISM值
        """
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        Rs = sobel(R)
        Gs = sobel(G)
        Bs = sobel(B)

        R_edge = np.multiply(Rs, R)
        G_edge = np.multiply(Gs, G)
        B_edge = np.multiply(Bs, B)

        r_eme = self.eme(R_edge, window_size)
        g_eme = self.eme(G_edge, window_size)
        b_eme = self.eme(B_edge, window_size)

        # 根据人眼视觉系统的RGB通道权重
        lambda_r = 0.299
        lambda_g = 0.587
        lambda_b = 0.114

        uism_value = lambda_r * r_eme + lambda_g * g_eme + lambda_b * b_eme
        return uism_value

    def eme(self, img, window_size):
        """
        计算EME值

        参数：
        - img: 输入的二维数组
        - window_size: 分块大小

        返回：
        - eme_value: 计算得到的EME值
        """
        img_height, img_width = img.shape
        k1 = img_width // window_size
        k2 = img_height // window_size
        w = 2.0 / (k1 * k2)
        eme_value = 0.0

        for i in range(k2):
            for j in range(k1):
                block = img[i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size]
                I_max = np.max(block)
                I_min = np.min(block)
                if I_min == 0:
                    continue
                eme_value += np.log(I_max / I_min)
        eme_value *= w
        return eme_value

    def compute_uiconm(self, img, window_size=10):
        """
        计算对比度测量指标 UIConM

        参数：
        - img: 输入的RGB图像，numpy数组
        - window_size: 分块大小，默认10

        返回：
        - uiconm_value: 计算得到的UIConM值
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_height, img_width = img_gray.shape
        k1 = img_width // window_size
        k2 = img_height // window_size
        w = -1.0 / (k1 * k2)
        uiconm_value = 0.0
        alpha = 1.0  # 指数，可以调整

        for i in range(k2):
            for j in range(k1):
                block = img_gray[i * window_size:(i + 1) * window_size, j * window_size:(j + 1) * window_size]
                I_max = np.max(block)
                I_min = np.min(block)
                top = I_max - I_min
                bot = I_max + I_min
                if bot == 0 or top == 0:
                    continue
                val = (top / bot) ** alpha * np.log(top / bot)
                uiconm_value += val
        uiconm_value *= w
        return uiconm_value


def main():
    # 示例使用
    image_folders = ['enhanced']  # 输入的图像文件夹列表，可以修改
    output_csv = 'UCIQE_UIQM.csv'  # 输出CSV文件路径，可以修改
    image_size = (640, 640)  # 图像预处理大小，可以修改
    uciqe_coeffs = (0.4680, 0.2745, 0.2576)  # UCIQE指标的系数，可以修改
    uiqm_coeffs = (0.0282, 0.2953, 3.5753)  # UIQM指标的系数，可以修改
    metrics_to_compute = ('uciqe', 'uiqm')  # 要计算的指标，可以修改

    uciqe_uiqm = UCIQE_UIQM(
        image_folders=image_folders,
        output_csv=output_csv,
        image_size=image_size,
        uciqe_coeffs=uciqe_coeffs,
        uiqm_coeffs=uiqm_coeffs,
        metrics_to_compute=metrics_to_compute
    )
    uciqe_uiqm.process_images()


if __name__ == '__main__':
    main()