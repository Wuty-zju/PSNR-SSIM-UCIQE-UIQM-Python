import numpy as np
import os
import csv
from skimage import io, color, filters
from skimage.metrics import structural_similarity
from skimage.transform import resize
from tqdm import tqdm
import time


class PSNR_SSIM_UIQM_UCIQE:
    """
    图像质量评估类，提供计算PSNR、SSIM、UIQM和UCIQE的功能。
    """

    def __init__(self, image_path, reference_path=None):
        """
        初始化方法。

        参数:
            image_path (str): 已处理图像的文件夹路径。
            reference_path (str, 可选): 原始参考图像的文件夹路径。
        """
        self.image_path = image_path
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

        # 确定窗口大小
        win_size = min(img1.shape[0], img1.shape[1], 7)
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3

        # 计算SSIM
        ssim_value = structural_similarity(img1, img2, channel_axis=-1, data_range=1.0, win_size=win_size)
        return ssim_value

    def compute_uiqm(self, img, m1=0.0282, m2=0.2953, m3=3.5753):
        """
        计算单张图像的UIQM。

        参数:
            img (numpy.ndarray): 输入图像。
            m1 (float): UICM的系数，默认值为0.0282。
            m2 (float): UISM的系数，默认值为0.2953。
            m3 (float): UIConM的系数，默认值为3.5753。

        返回:
            float: 计算得到的UIQM值。
        """
        # 将图像转换为浮点型，范围[0, 1]
        rgb = img.astype(np.float32) / 255.0

        # 计算UICM
        uicm = self._calculate_uicm(rgb)

        # 计算UISM
        uism = self._calculate_uism(rgb)

        # 计算UIConM
        gray = color.rgb2gray(rgb)
        uiconm = self._calculate_uiconm(gray)

        # 组合计算UIQM
        uiqm_value = m1 * uicm + m2 * uism + m3 * uiconm
        return uiqm_value

    def compute_uciqe(self, img, c1=0.4680, c2=0.2745, c3=0.2576):
        """
        计算单张图像的UCIQE。

        参数:
            img (numpy.ndarray): 输入图像。
            c1 (float): 第一个系数，默认值为0.4680。
            c2 (float): 第二个系数，默认值为0.2745。
            c3 (float): 第三个系数，默认值为0.2576。

        返回:
            float: 计算得到的UCIQE值。
        """
        # 将图像转换为浮点型，范围[0, 1]
        rgb = img.astype(np.float32) / 255.0
        lab = color.rgb2lab(rgb)

        # 提取L、a、b分量，并归一化
        L = lab[:, :, 0] / 100.0  # L* 范围 [0, 1]
        a = lab[:, :, 1] / 128.0  # a* 近似范围 [-1, 1]
        b = lab[:, :, 2] / 128.0  # b* 近似范围 [-1, 1]

        # 计算色度C = sqrt(a^2 + b^2)
        chroma = np.sqrt(a ** 2 + b ** 2)

        # 计算σ_c（色度的标准差）
        sigma_c = np.std(chroma)

        # 计算亮度对比度 con_l = 第99百分位 - 第1百分位
        con_l = np.percentile(L, 99) - np.percentile(L, 1)

        # 计算饱和度 S = arccos(L / sqrt(L^2 + chroma^2))
        epsilon = 1e-8  # 防止除零
        denominator = np.sqrt(L ** 2 + chroma ** 2) + epsilon
        # 确保除法结果在 [-1, 1] 范围内
        cos_theta = np.clip(L / denominator, -1, 1)
        saturation = np.arccos(cos_theta)

        # 计算饱和度的平均值
        mean_saturation = np.mean(saturation)

        # 组合计算UCIQE
        uciqe_value = c1 * sigma_c + c2 * con_l + c3 * mean_saturation
        return uciqe_value

    def _calculate_uicm(self, rgb):
        """
        内部方法，计算UICM。

        参数:
            rgb (numpy.ndarray): 输入RGB图像。

        返回:
            float: 计算得到的UICM值。
        """
        # 提取R、G、B通道
        R = rgb[:, :, 0]
        G = rgb[:, :, 1]
        B = rgb[:, :, 2]

        # 计算RG和YB通道
        RG = R - G
        YB = 0.5 * (R + G) - B

        # 计算经过α裁剪的均值和标准差
        alpha = 0.1  # α裁剪比例，可根据需要调整
        mu_RG = self._asymmetric_trimmed_mean(RG, alpha, alpha)
        mu_YB = self._asymmetric_trimmed_mean(YB, alpha, alpha)
        sigma_RG = self._asymmetric_trimmed_std(RG, alpha, alpha)
        sigma_YB = self._asymmetric_trimmed_std(YB, alpha, alpha)

        # 计算UICM
        c1 = -0.0268
        c2 = 0.1586
        uicm_value = c1 * np.sqrt(mu_RG ** 2 + mu_YB ** 2) + c2 * np.sqrt(sigma_RG ** 2 + sigma_YB ** 2)
        return uicm_value

    def _calculate_uism(self, rgb):
        """
        内部方法，计算UISM。

        参数:
            rgb (numpy.ndarray): 输入RGB图像。

        返回:
            float: 计算得到的UISM值。
        """
        # 对每个通道进行Sobel边缘检测
        R = rgb[:, :, 0]
        G = rgb[:, :, 1]
        B = rgb[:, :, 2]

        R_sobel = filters.sobel(R)
        G_sobel = filters.sobel(G)
        B_sobel = filters.sobel(B)

        # 将边缘图与原始通道相乘，得到加权边缘图
        R_edge = R_sobel * R
        G_edge = G_sobel * G
        B_edge = B_sobel * B

        # 对加权边缘图计算EME
        R_eme = self._eme(R_edge)
        G_eme = self._eme(G_edge)
        B_eme = self._eme(B_edge)

        # 组合计算UISM
        lambda_r = 0.299
        lambda_g = 0.587
        lambda_b = 0.114
        uism_value = lambda_r * R_eme + lambda_g * G_eme + lambda_b * B_eme
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
        # 防止对数计算中的除零
        channel = channel + 1e-8
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
                block_max = max(block_max, block_min + 1e-8)  # 确保block_max > block_min

                eme_value += w * np.log(block_max / block_min)

        return eme_value

    def _asymmetric_trimmed_mean(self, arr, alpha_low, alpha_high):
        """
        计算数组的非对称裁剪平均值。

        参数:
            arr (numpy.ndarray): 输入数组。
            alpha_low (float): 低端裁剪比例，取值范围[0, 0.5)。
            alpha_high (float): 高端裁剪比例，取值范围[0, 0.5)。

        返回:
            float: 裁剪后的平均值。
        """
        arr_flat = arr.flatten()
        arr_sorted = np.sort(arr_flat)
        N = len(arr_sorted)
        lower_idx = int(np.floor(N * alpha_low))
        upper_idx = int(np.ceil(N * (1 - alpha_high)))

        arr_cropped = arr_sorted[lower_idx:upper_idx]
        mean_value = np.mean(arr_cropped)
        return mean_value

    def _asymmetric_trimmed_std(self, arr, alpha_low, alpha_high):
        """
        计算数组的非对称裁剪标准差。

        参数:
            arr (numpy.ndarray): 输入数组。
            alpha_low (float): 低端裁剪比例，取值范围[0, 0.5)。
            alpha_high (float): 高端裁剪比例，取值范围[0, 0.5)。

        返回:
            float: 裁剪后的标准差。
        """
        arr_flat = arr.flatten()
        arr_sorted = np.sort(arr_flat)
        N = len(arr_sorted)
        lower_idx = int(np.floor(N * alpha_low))
        upper_idx = int(np.ceil(N * (1 - alpha_high)))

        arr_cropped = arr_sorted[lower_idx:upper_idx]
        std_value = np.std(arr_cropped)
        return std_value

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
        # 转换回[0,255]范围，防止除零
        channel = channel * 255.0 + 1e-8
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

                # 防止除零
                block_min = max(block_min, 1e-8)
                block_max = max(block_max, 1e-8)

                top = self._plipsub(block_max, block_min, gamma)
                bottom = self._plipsum(block_max, block_min, gamma)
                m = top / (bottom + 1e-8)  # 防止除零

                if m > 0:
                    s += m * np.log(m)

        return self._plipmult(w, s, gamma)

    def process_images(self, metrics=['psnr', 'ssim', 'uiqm', 'uciqe'], csv_path="metrics.csv",
                       uiqm_coeffs=(0.0282, 0.2953, 3.5753), uciqe_coeffs=(0.4680, 0.2745, 0.2576),
                       target_size=(640, 640)):
        """
        处理图像并计算指定的指标。

        参数:
            metrics (list of str): 要计算的指标列表。
            csv_path (str): 保存CSV的相对路径和文件名。
            uiqm_coeffs (tuple): UIQM的系数，默认值为(0.0282, 0.2953, 3.5753)。
            uciqe_coeffs (tuple): UCIQE的系数，默认值为(0.4680, 0.2745, 0.2576)。
            target_size (tuple of int): 图像缩放的目标尺寸，默认值为(640, 640)。

        返回:
            None
        """
        # 获取图像文件列表，仅包括常见图像格式
        result_files = [f for f in os.listdir(self.image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        N = len(result_files)

        for img_file in tqdm(result_files, desc="Processing images"):
            # 已处理图像
            corrected_path = os.path.join(self.image_path, img_file)
            corrected = io.imread(corrected_path)

            # 缩放图像，如果指定了target_size
            if target_size is not None:
                corrected = resize(corrected, target_size, anti_aliasing=True, preserve_range=True)
                corrected = corrected.astype(np.uint8)

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
                    # 缩放参考图像，如果指定了target_size
                    if target_size is not None:
                        reference = resize(reference, target_size, anti_aliasing=True, preserve_range=True)
                        reference = reference.astype(np.uint8)
                    if 'psnr' in metrics:
                        start_time = time.perf_counter()
                        psnr_value = self.compute_psnr(corrected, reference)
                        psnr_time = time.perf_counter() - start_time
                        result['psnr_value'] = psnr_value
                        result['psnr_time'] = psnr_time
                    if 'ssim' in metrics:
                        try:
                            start_time = time.perf_counter()
                            ssim_value = self.compute_ssim(corrected, reference)
                            ssim_time = time.perf_counter() - start_time
                            result['ssim_value'] = ssim_value
                            result['ssim_time'] = ssim_time
                        except ValueError as e:
                            print(f"计算 {img_file} 的 SSIM 时出错：{e}")
                            result['ssim_value'] = None
                            result['ssim_time'] = None
            else:
                if 'psnr' in metrics or 'ssim' in metrics:
                    print("未提供参考路径，无法计算 PSNR 和 SSIM。")

            # 计算非参考指标
            if 'uiqm' in metrics:
                start_time = time.perf_counter()
                uiqm_value = self.compute_uiqm(corrected, m1=uiqm_coeffs[0], m2=uiqm_coeffs[1], m3=uiqm_coeffs[2])
                uiqm_time = time.perf_counter() - start_time
                result['uiqm_value'] = uiqm_value
                result['uiqm_time'] = uiqm_time
            if 'uciqe' in metrics:
                start_time = time.perf_counter()
                uciqe_value = self.compute_uciqe(corrected, c1=uciqe_coeffs[0], c2=uciqe_coeffs[1], c3=uciqe_coeffs[2])
                uciqe_time = time.perf_counter() - start_time
                result['uciqe_value'] = uciqe_value
                result['uciqe_time'] = uciqe_time

            # 保存结果
            self.results.append(result)

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

        # 根据指标构建字段名
        fieldnames = ['image_name', 'image_path']
        for metric in metrics:
            fieldnames.append(metric + '_value')
            fieldnames.append(metric + '_time')

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in self.results:
                # 确保所有字段都存在
                row = {key: result.get(key, '') for key in fieldnames}
                writer.writerow(row)


if __name__ == '__main__':
    # 定义结果图像和参考图像的文件夹路径
    image_path = 'enhanced'       # 已处理图像文件夹路径
    reference_path = 'origin'     # 参考图像文件夹路径

    # 创建图像质量评估对象
    psuu = PSNR_SSIM_UIQM_UCIQE(image_path, reference_path)

    # 指定要计算的指标
    metrics_to_compute = [#'psnr', 'ssim', 
                          'uiqm', 'uciqe']

    # 指定UIQM和UCIQE的系数
    uiqm_coeffs = (0.0282, 0.2953, 3.5753)
    uciqe_coeffs = (0.4680, 0.2745, 0.2576)

    # 处理图像并计算指标，默认将图像缩放到640×640大小
    psuu.process_images(metrics=metrics_to_compute, csv_path="PSNR_SSIM_UIQM_UCIQE.csv",
                        uiqm_coeffs=uiqm_coeffs, uciqe_coeffs=uciqe_coeffs, target_size=(640, 640))