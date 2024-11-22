"""
用于水下图像质量评估的指标。

描述：
该模块定义了一个用于计算图像质量指标的类，包括PSNR、SSIM、UIQM和UCIQE。
可以选择性地计算这些指标，支持批量处理图像并将结果保存到CSV文件中。
"""

import numpy as np
import os
import csv
from skimage import io, color, filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class PSNR_SSIM_UIQM_UCIQE:
    """
    图像质量评估类，提供计算PSNR、SSIM、UIQM和UCIQE的功能。
    """

    def __init__(self, result_path, reference_path=None):
        """
        初始化方法。

        参数:
            result_path (str): 已处理图像的文件夹路径。
            reference_path (str, 可选): 原始参考图像的文件夹路径。
        """
        self.result_path = result_path
        self.reference_path = reference_path
        self.results = []

    def compute_psnr(self, img1, img2):
        """
        计算两幅图像之间的PSNR。

        参数:
            img1 (numpy.ndarray): 图像1（已处理图像）。
            img2 (numpy.ndarray): 图像2（参考图像）。

        返回:
            float: 计算得到的PSNR值。如果图像完全相同，则返回 float('inf')。
        """
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        mse = np.mean((img1 - img2) ** 2)

        if mse == 0:  # 图像完全一致
            return float('inf')
        psnr_value = 10 * np.log10(1 / mse)
        return psnr_value

    def compute_ssim(self, img1, img2):
        """
        计算两幅图像之间的SSIM。

        参数:
            img1 (numpy.ndarray): 图像1（已处理图像）。
            img2 (numpy.ndarray): 图像2（参考图像）。

        返回:
            float: 计算得到的SSIM值。
        """
        # 将图像转换为浮点型，范围[0, 1]
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        # 确定win_size
        win_size = min(img1.shape[0], img1.shape[1], 7)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3

        # 计算SSIM
        ssim_value = structural_similarity(img1, img2, channel_axis=-1, data_range=1.0, win_size=win_size)
        return ssim_value

    def compute_uiqm(self, img):
        """
        计算单张图像的UIQM。

        参数:
            img (numpy.ndarray): 输入图像。

        返回:
            float: 计算得到的UIQM值。
        """
        # 将图像转换为浮点型，范围[0, 1]
        rgb = img.astype(np.float32) / 255.0

        # 参数设置
        p1, p2, p3 = 0.0282, 0.2953, 3.5753

        # 计算UICM
        uicm = self._calculate_uicm(rgb)

        # 计算UISM
        uism = self._calculate_uism(rgb)

        # 计算UIConM
        gray = color.rgb2gray(rgb)
        uiconm = self._calculate_uiconm(gray)

        # 组合计算UIQM
        uiqm_value = p1 * uicm + p2 * uism + p3 * uiconm
        return uiqm_value

    def compute_uciqe(self, img):
        """
        计算单张图像的UCIQE。

        参数:
            img (numpy.ndarray): 输入图像。

        返回:
            float: 计算得到的UCIQE值。
        """
        # 将图像转换为浮点型，范围[0, 1]
        rgb = img.astype(np.float32) / 255.0
        lab = color.rgb2lab(rgb)

        # 参数设置
        c1, c2, c3 = 0.4680, 0.2745, 0.2576

        # 提取L、a、b分量
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        # 计算色度
        chroma = np.sqrt(a ** 2 + b ** 2)
        sc = np.std(chroma)

        # 计算亮度对比度
        conl = np.percentile(L, 99) - np.percentile(L, 1)

        # 计算平均饱和度
        saturation = chroma / (L + 1e-8)  # 防止除零
        us = np.mean(saturation)

        # 组合计算UCIQE
        uciqe_value = c1 * sc + c2 * conl + c3 * us
        return uciqe_value

    def _calculate_uicm(self, rgb):
        """
        内部方法，计算UICM。

        参数:
            rgb (numpy.ndarray): 输入RGB图像。

        返回:
            float: 计算得到的UICM值。
        """
        # 计算RG和YB
        rg = rgb[:, :, 0] - rgb[:, :, 1]
        yb = (rgb[:, :, 0] + rgb[:, :, 1]) / 2 - rgb[:, :, 2]

        # 计算均值和方差
        urg = np.mean(rg)
        s2rg = np.var(rg)
        uyb = np.mean(yb)
        s2yb = np.var(yb)

        # 计算UICM
        uicm_value = -0.0268 * np.sqrt(urg ** 2 + uyb ** 2) + 0.1586 * np.sqrt(s2rg + s2yb)
        return uicm_value

    def _calculate_uism(self, rgb):
        """
        内部方法，计算UISM。

        参数:
            rgb (numpy.ndarray): 输入RGB图像。

        返回:
            float: 计算得到的UISM值。
        """
        # 计算每个通道的Sobel梯度
        R_sobel = filters.sobel(rgb[:, :, 0])
        G_sobel = filters.sobel(rgb[:, :, 1])
        B_sobel = filters.sobel(rgb[:, :, 2])

        # 计算EME
        R_eme = self._eme(R_sobel)
        G_eme = self._eme(G_sobel)
        B_eme = self._eme(B_sobel)

        # 组合计算UISM
        uism_value = 0.299 * R_eme + 0.587 * G_eme + 0.114 * B_eme
        return uism_value

    def _calculate_uiconm(self, gray):
        """
        内部方法，计算UIConM。

        参数:
            gray (numpy.ndarray): 输入灰度图像。

        返回:
            float: 计算得到的UIConM值。
        """
        uiconm_value = self._logamee(gray)
        return uiconm_value

    def _eme(self, channel, block_size=8):
        """
        计算EME（增强测量估计）。

        参数:
            channel (numpy.ndarray): 图像的单个通道。
            block_size (int): 分块大小。

        返回:
            float: 计算得到的EME值。
        """
        channel = channel + 1e-8  # 防止对数计算中的除零
        M, N = channel.shape
        num_x = int(np.ceil(M / block_size))
        num_y = int(np.ceil(N / block_size))
        eme_value = 0.0
        w = 2.0 / (num_x * num_y)

        for i in range(num_x):
            for j in range(num_y):
                x_start = i * block_size
                x_end = min((i + 1) * block_size, M)
                y_start = j * block_size
                y_end = min((j + 1) * block_size, N)

                block = channel[x_start:x_end, y_start:y_end]
                block_min = np.min(block)
                block_max = np.max(block)

                # 防止对数计算中的除零
                block_min = max(block_min, 1e-8)
                block_max = max(block_max, 1e-8)

                eme_value += w * np.log(block_max / block_min)

        return eme_value

    def _plipsum(self, i, j, gamma=1026):
        """
        PLIP模型的加法运算。

        参数:
            i (float): 值i。
            j (float): 值j。
            gamma (float): PLIP模型的gamma参数。

        返回:
            float: 计算结果。
        """
        return i + j - (i * j) / gamma

    def _plipsub(self, i, j, gamma=1026):
        """
        PLIP模型的减法运算。

        参数:
            i (float): 值i。
            j (float): 值j。
            gamma (float): PLIP模型的gamma参数。

        返回:
            float: 计算结果。
        """
        return gamma * (i - j) / (gamma - j + 1e-8)  # 防止除零

    def _plipmult(self, c, j, gamma=1026):
        """
        PLIP模型的乘法运算。

        参数:
            c (float): 常数c。
            j (float): 值j。
            gamma (float): PLIP模型的gamma参数。

        返回:
            float: 计算结果。
        """
        return gamma - gamma * (1 - j / gamma) ** c

    def _logamee(self, channel, block_size=8):
        """
        计算对数AMBE（平均亮度误差）。

        参数:
            channel (numpy.ndarray): 灰度图像。
            block_size (int): 分块大小。

        返回:
            float: 计算得到的对数AMBE值。
        """
        channel = channel * 255.0 + 1e-8  # 转换回[0,255]范围，防止除零
        M, N = channel.shape
        num_x = int(np.ceil(M / block_size))
        num_y = int(np.ceil(N / block_size))
        s = 0.0
        w = 1.0 / (num_x * num_y)
        gamma = 1026

        for i in range(num_x):
            for j in range(num_y):
                x_start = i * block_size
                x_end = min((i + 1) * block_size, M)
                y_start = j * block_size
                y_end = min((j + 1) * block_size, N)

                block = channel[x_start:x_end, y_start:y_end]
                block_min = np.min(block)
                block_max = np.max(block)

                top = self._plipsub(block_max, block_min, gamma)
                bottom = self._plipsum(block_max, block_min, gamma)
                m = top / (bottom + 1e-8)  # 防止除零

                if m > 0:
                    s += m * np.log(m)

        return self._plipmult(w, s, gamma)

    def process_images(self, metrics=['psnr', 'ssim', 'uiqm', 'uciqe'], csv_path="metrics.csv"):
        """
        处理图像并计算指定的指标。

        参数:
            metrics (list of str): 要计算的指标列表，默认为所有指标。
            csv_path (str): 保存CSV的相对路径和文件名。

        返回:
            None
        """
        result_files = [f for f in os.listdir(self.result_path)]
        N = len(result_files)
        sum_metrics = {metric: 0.0 for metric in metrics}

        for img_file in result_files:
            # 已处理图像
            corrected_path = os.path.join(self.result_path, img_file)
            corrected = io.imread(corrected_path)

            # 添加 image_path 列，值为相对于保存的 CSV 文件的路径
            result = {
                'image_name': img_file,
                'image_path': os.path.relpath(corrected_path, start=os.path.dirname(csv_path)).split(os.sep)[0]
            }

            # 如果需要计算参考指标，确保提供了参考路径
            if self.reference_path and ('psnr' in metrics or 'ssim' in metrics):
                reference_path = os.path.join(self.reference_path, img_file)
                if not os.path.exists(reference_path):
                    print(f"参考图像 {img_file} 不存在，跳过 PSNR 和 SSIM 计算。")
                else:
                    reference = io.imread(reference_path)
                    if 'psnr' in metrics:
                        psnr_value = self.compute_psnr(corrected, reference)
                        result['psnr'] = psnr_value
                        sum_metrics['psnr'] += psnr_value
                    if 'ssim' in metrics:
                        try:
                            ssim_value = self.compute_ssim(corrected, reference)
                            result['ssim'] = ssim_value
                            sum_metrics['ssim'] += ssim_value
                        except ValueError as e:
                            print(f"计算 {img_file} 的 SSIM 时出错：{e}")
                            result['ssim'] = None
            else:
                if 'psnr' in metrics or 'ssim' in metrics:
                    print("未提供参考路径，无法计算 PSNR 和 SSIM。")

            # 计算非参考指标
            if 'uiqm' in metrics:
                uiqm_value = self.compute_uiqm(corrected)
                result['uiqm'] = uiqm_value
                sum_metrics['uiqm'] += uiqm_value
            if 'uciqe' in metrics:
                uciqe_value = self.compute_uciqe(corrected)
                result['uciqe'] = uciqe_value
                sum_metrics['uciqe'] += uciqe_value

            # 保存结果
            self.results.append(result)
        """
        # 计算平均值
        avg_metrics = {metric: sum_metrics[metric] / N for metric in metrics if N > 0}
        avg_metrics['image_name'] = 'Average'
        avg_metrics['image_path'] = ""
        self.results.append(avg_metrics)
        """

        # 将结果保存为CSV文件
        self._save_results_to_csv(metrics, csv_path)
        
    def _save_results_to_csv(self, metrics, csv_path):
        """
        将结果保存到CSV文件。

        参数:
            metrics (list of str): 要保存的指标列表。
            csv_path (str): 保存CSV的相对路径和文件名。

        返回:
            None
        """
        output_file = os.path.join(os.path.dirname(__file__), csv_path)
        fieldnames = ['image_name', 'image_path'] + metrics

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                # 确保所有字段都存在
                row = {key: result.get(key, '') for key in fieldnames}
                writer.writerow(row)

if __name__ == '__main__':
    # 定义结果图像和参考图像的文件夹路径
    result_path = 'enhanced'       # 替换为实际的已处理图像文件夹路径
    reference_path = 'origin'      # 替换为实际的参考图像文件夹路径

    # 创建图像质量评估对象
    iqm = PSNR_SSIM_UIQM_UCIQE(result_path, reference_path)

    # 指定要计算的指标，可以根据需要调整
    metrics_to_compute = ['psnr', 'ssim', 'uiqm', 'uciqe']

    # 处理图像并计算指标
    iqm.process_images(metrics=metrics_to_compute, csv_path="PSNR_SSIM_UIQM_UCIQE.csv")