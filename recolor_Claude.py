"""
图像重新着色模块
================

核心算法：反距离加权插值（Inverse Distance Weighting, IDW）
原理：根据像素颜色与调色板颜色的距离，加权混合多个调色板颜色的变换，
      实现平滑的颜色过渡效果，避免产生突兀的色块。

数学基础：
对于每个像素，其新颜色由所有调色板颜色的变换贡献加权得到：
I_out = Σ(w_i × (I_in + C_tgt_i - C_src_i))
其中 w_i = 1/d_i / Σ(1/d_j)，d_i 是像素到第i个调色板颜色的距离
"""

import numpy as np
from skimage.color import rgb2lab, lab2rgb


def rgb_transfer(Iin, C_src, C_tgt, mask=None):
    """
    在RGB颜色空间进行颜色迁移

    参数：
        Iin: 输入图像，RGB格式，值域[0,255]
        C_src: 源调色板，Lab格式，shape=(k,3)，k为调色板颜色数
        C_tgt: 目标调色板，Lab格式，shape=(k,3)
        mask: 二值掩码，指定要处理的区域，1表示处理，0表示保持原样

    返回：
        Iout: 重新着色后的图像，RGB格式，值域[0,1]

    算法流程：
    1. 将调色板从Lab转换到RGB（因为要在RGB空间进行插值）
    2. 将图像展平为二维数组便于向量化计算
    3. 使用反距离加权插值计算新颜色
    4. 恢复图像形状
    """

    # 将输入图像归一化到[0,1]范围，便于后续计算
    Iin = np.array(Iin / 255.).astype(np.float32)

    # 将调色板从Lab空间转换到RGB空间
    # 注意：lab2rgb需要3维输入，所以先扩展维度再压缩
    C_src = lab2rgb(np.expand_dims(C_src, axis=0))
    C_tgt = lab2rgb(np.expand_dims(C_tgt, axis=0))
    C_src = np.squeeze(C_src, axis=0)
    C_tgt = np.squeeze(C_tgt, axis=0)

    # 如果没有提供掩码，默认处理整个图像
    if mask is None:
        mask = np.ones_like(Iin[:, :, 0])

    # 获取图像尺寸：m×n像素，b个通道（RGB为3）
    m, n, b = Iin.shape

    # 将图像从3D(m,n,b)展平为2D(m*n,b)，便于向量化处理
    # 每一行代表一个像素的RGB值
    Iin = np.reshape(Iin, (m * n, b))
    Iout = Iin.copy()  # 创建输出图像的副本
    mask = mask.flatten()  # 将掩码也展平为1D数组

    # 执行颜色迁移的核心算法
    Iout = ab_transfer(Iin, C_src, C_tgt, mask=mask)

    # 将结果重新整形为原始图像尺寸
    Iout = np.reshape(Iout, (m, n, b))

    return Iout


def lab_transfer(Iin, C_src, C_tgt, mask=None, mode=2):
    """
    在Lab颜色空间进行颜色迁移（推荐方法）

    参数：
        Iin: 输入图像，Lab格式
        C_src: 源调色板，Lab格式
        C_tgt: 目标调色板，Lab格式
        mask: 处理区域掩码
        mode: 迁移模式
              2 - 只迁移ab通道（保持亮度L不变，只改变颜色）
              3 - 迁移所有Lab通道（同时改变亮度和颜色）

    返回：
        Iout: 重新着色的图像，RGB格式，值域[0,1]
        Pout: 目标调色板的副本

    Lab空间的优势：
    - L通道（亮度）与ab通道（颜色）分离
    - 可以独立控制亮度和颜色的变化
    - 避免颜色调整影响图像的明暗细节
    """

    # 如果没有提供掩码，默认处理整个图像
    if mask is None:
        mask = np.ones_like(Iin[:, :, 0])

    # 保存目标调色板的副本（用于可能的后续处理）
    Pout = C_tgt.copy()

    # 获取图像尺寸
    m, n, b = Iin.shape

    # 将图像展平为2D数组：从(m,n,3)变为(m*n,3)
    # 注意：输入已经是Lab格式，无需转换
    Iin = np.reshape(Iin, (m * n, b))
    Iout = Iin.copy()  # 创建输出副本
    mask = mask.flatten()  # 展平掩码

    # 注意：调色板已经是Lab格式，无需转换
    # 下面注释的代码是从RGB转换的情况
    # C_src = rgb2lab(np.expand_dims(C_src, axis=0))
    # C_tgt = rgb2lab(np.expand_dims(C_tgt, axis=0))
    # C_src = np.squeeze(C_src, axis=0)
    # C_tgt = np.squeeze(C_tgt, axis=0)

    if mode == 2:
        # 模式2：只迁移ab通道（颜色），保持L通道（亮度）不变
        # 这是最常用的模式，可以改变颜色而不影响图像的明暗细节
        # Iin[:, 1:]提取ab通道，C_src[:, 1:]提取调色板的ab分量
        Iout[:, 1:] = ab_transfer(Iin[:, 1:], C_src[:, 1:], C_tgt[:, 1:], mask=mask)
    else:
        # 模式3：迁移所有通道，包括亮度
        # 这会同时改变图像的颜色和明暗，可能产生更戏剧化的效果
        Iout[:, 0:] = ab_transfer(Iin[:, 0:], C_src[:, 0:], C_tgt[:, 0:], mask=mask)

    # 将结果重新整形为原始尺寸
    Iout = np.reshape(Iout, (m, n, b))

    # 将Lab格式转换回RGB格式用于显示和保存
    Iout = lab2rgb(Iout)

    return Iout, Pout


def ab_transfer(I_src, C_src, C_tgt, mask=None):
    """
    核心颜色迁移算法 - 反距离加权插值

    参数：
        I_src: 源图像像素值，shape=(m,b)，m为像素数，b为通道数
        C_src: 源调色板，shape=(k,b)，k为调色板颜色数
        C_tgt: 目标调色板，shape=(k,b)
        mask: 处理掩码，shape=(m,)

    返回：
        I_tgt: 迁移后的像素值

    算法原理：
    反距离加权（IDW）插值是一种空间插值方法，核心思想是：
    1. 距离越近的点影响越大（权重越高）
    2. 权重与距离成反比：w = 1/d
    3. 最终结果是所有点贡献的加权平均

    在颜色迁移中的应用：
    - 每个像素根据其与各调色板颜色的距离获得不同权重
    - 权重决定了每个调色板颜色变换对该像素的贡献
    - 这确保了平滑的颜色过渡，避免色块效应
    """

    # 如果没有掩码，处理所有像素
    if mask is None:
        mask = np.ones_like(I_src[:, 0])

    # 初始化输出数组
    I_tgt = np.zeros_like(I_src)
    [m, b] = I_src.shape  # m个像素，b个通道

    # ========== 计算权重矩阵 ==========
    # W[i,j]表示第i个像素对第j个调色板颜色的权重

    k = np.size(C_src, 0)  # 调色板中的颜色数量
    eps = 0.0001  # 防止除零的小常数
    W = np.zeros((m, k))  # 权重矩阵：m个像素 × k个调色板颜色

    # 对每个调色板颜色计算权重
    for i in range(k):
        # 计算每个像素到第i个调色板颜色的欧氏距离平方
        D = np.zeros(m)
        for j in range(b):
            # 累加各通道的差值平方：D = Σ(I_src - C_src)²
            D = D + (I_src[:, j] - C_src[i, j]) ** 2

        # 反距离权重：w = 1/(d+eps)
        # eps避免当像素颜色与调色板颜色完全相同时出现除零错误
        W[:, i] = 1. / (D + eps)

    # print(k,b)  # 调试信息：显示调色板大小和通道数

    # ========== 归一化权重 ==========
    # 确保每个像素的所有权重和为1（概率分布）
    sumW = np.sum(W, 1)  # 计算每个像素的权重总和
    for j in range(k):
        W[:, j] = W[:, j] / sumW  # 归一化：w_i = w_i / Σw_j

    # ========== 应用颜色迁移 ==========
    # 核心公式：I_tgt = Σ[w_i × (I_src + ΔC_i)]
    # 其中 ΔC_i = C_tgt_i - C_src_i 是第i个调色板颜色的变化量

    for i in range(k):  # 对每个调色板颜色
        for j in range(b):  # 对每个通道
            # 累加加权贡献：每个调色板颜色的变换按权重贡献到最终结果
            # I_src[:, j]：原始像素值
            # C_tgt[i, j] - C_src[i, j]：调色板颜色的变化量
            # W[:, i]：该调色板颜色对各像素的权重
            I_tgt[:, j] = I_tgt[:, j] + W[:, i] * (I_src[:, j] + C_tgt[i, j] - C_src[i, j])

    # ========== 应用掩码 ==========
    # 将掩码外的像素保持原样（不进行颜色迁移）
    idx = np.argwhere(mask == 0)  # 找到掩码为0的像素索引
    I_tgt[idx, :] = I_src[idx, :]  # 恢复原始颜色

    return I_tgt


def lab_transfer_cls(Iin, C_src, C_tgt, mask=None, valid_class=None):
    """
    基于语义类别的Lab颜色迁移（高级功能）

    参数：
        Iin: 输入图像，RGB格式
        C_src: 每个类别的源调色板，shape=(num_class, k, 3)
        C_tgt: 每个类别的目标调色板
        mask: 语义分割掩码，每个像素的类别标签
        valid_class: 有效类别列表（要处理的类别）

    返回：
        Iout: 重新着色的图像
        Pout: 目标调色板

    应用场景：
    当图像包含不同语义区域（如天空、草地、建筑）时，
    可以为每个区域使用不同的调色板进行独立的颜色调整。

    例如：
    - 天空使用蓝色系调色板
    - 草地使用绿色系调色板
    - 建筑使用暖色系调色板
    """

    # 初始化掩码
    if mask is None:
        mask = np.ones_like(Iin[:, :, 0])

    Pout = C_tgt.copy()
    m, n, b = Iin.shape

    # 颜色空间转换：RGB -> Lab
    Iin = rgb2lab(Iin)
    Iin = np.reshape(Iin, (m * n, b))
    Iout = Iin.copy()  # 最终输出
    Iout_cls = Iin.copy()  # 临时存储单个类别的处理结果

    mask = mask.flatten()
    mask_bin = np.zeros_like(mask)  # 二值化掩码，用于标记当前处理的类别

    # 对每个有效类别分别进行颜色迁移
    for id_cls, cls in enumerate(valid_class):
        # 跳过没有调色板的类别
        if C_src[id_cls] == 0:
            continue

        # 创建当前类别的二值掩码
        mask_bin[mask == cls] = 1

        # 对当前类别的像素进行颜色迁移（只改变ab通道）
        Iout_cls[:, 1:] = ab_transfer(
            Iin[:, 1:],
            C_src[id_cls][:, 1:],
            C_tgt[id_cls][:, 1:],
            mask=mask_bin
        )

        # 将当前类别的处理结果更新到输出图像
        # 只更新属于当前类别的像素
        Iout[mask_bin, 1:] = Iout_cls[mask_bin, 1:]

    # 恢复图像形状并转换回RGB
    Iout = np.reshape(np.round(Iout), (m, n, b))
    Iout = lab2rgb(Iout)

    return Iout, Pout