import numpy as np
import pandas as pd
import heapq
import scipy
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb, rgb2hsv
from color_naming.color_naming import img2color_rgb, id_joost2label, id_ca2label, id_yu2label

# --- 全局数据加载与初始化 ---
# 加载 Word-to-Color (W2C) 查找表/数据库
# W2CRGB: 包含了大量 RGB 颜色点及其到基本颜色名称（如 11 个基本颜色名）的概率映射。
df2 = pd.read_csv('./color_naming/w2c_rgb.csv', header=None)
W2CRGB = df2.values  # 形状可能是 (N, 3) N个颜色点
# 将 RGB 颜色转换为 HSV 和 Lab 空间，方便后续在不同颜色空间进行聚类和计算。
W2CHSV = np.squeeze(rgb2hsv(np.expand_dims(W2CRGB / 255., axis=0)), axis=0)
W2CLAB = np.squeeze(rgb2lab(np.expand_dims(W2CRGB / 255., axis=0)), axis=0)


def img2color(img_lab, W2C, name_type='joost'):
    """
    将 Lab 颜色（通常是单个颜色点或调色板）转换为颜色名称标签和概率分布。

    基本原理：利用预先加载的 W2C 查找表（Word-to-Color mapping）进行颜色命名查询。

    Args:
        img_lab (np.ndarray): 输入颜色，Lab 空间 (N, 3)。
        W2C (np.ndarray): Word-to-Color 查找表。
        name_type (str): 使用的颜色命名模型类型 ('joost', 'ca', 'yu')。

    Returns:
        tuple: (颜色标签, 颜色名称, 颜色映射图, 概率分布图)
    """
    # 转换为 RGB 空间 (0-255) 以便调用 img2color_rgb 函数
    img_lab = np.expand_dims(img_lab, axis=0)
    img_rgb = lab2rgb(img_lab) * 255
    # 调用颜色命名模块进行查找
    color_label, color_nam, color_map, prob_map = img2color_rgb(img_rgb, W2C, name_type)

    return color_label, color_nam, color_map, prob_map


def color_clustering(cluster_num, color_percent, W2C, name_type='joost', extra_num=5, cluster_type='saturation',
                     colorspace='lab', mode=2):
    """
    为每个基本颜色名称（如 Red, Blue）提取一组最具代表性的原型颜色 (Prototype Colors)。

    算法步骤：
    1. 识别纯色：根据 W2C 表，找出对某个颜色名称具有最高概率的颜色点（即纯色）。
    2. 筛选：基于 `color_percent` 参数，筛选出 W2C 概率最高的颜色子集。
    3. K-Means 聚类：对筛选出的颜色子集进行 K-Means 聚类。聚类数量根据颜色名称是否为“附加色”进行调整。
    4. 原型选择：从每个聚类中心中，根据 `cluster_type`（饱和度、概率、概率+饱和度）选择一个最具代表性的颜色作为原型颜色。

    Args:
        cluster_num (int): 每个颜色名称的基础聚类数量。
        color_percent (float): 筛选纯色的百分比阈值。
        W2C (np.ndarray): Word-to-Color 查找表。
        cluster_type (str): 从聚类中选择原型颜色的标准 ('saturation', 'prob', 'probysat')。
        colorspace (str): 聚类操作使用的颜色空间 ('lab', 'rgb')。
        mode (int): Lab 空间聚类时使用的维度 (2 for ab-only, 3 for full Lab)。

    Returns:
        tuple: (原型颜色 RGB, 原型颜色 Lab, 原型颜色 HSV, 每个颜色名称的原型数量)
    """
    num, class_num = W2C.shape
    # w2cM: 每个颜色点被判定为哪个颜色名称（最高概率）
    w2cM = np.argmax(W2C, 1)

    sub_color_percent = 0.15  # 仅在 cluster_type='probysat' 时使用

    prototype_rgb, prototype_lab, prototype_hsv, num_prototype = [], [], [], []

    # 遍历所有颜色名称类别 (class_num 通常为 11)
    for label in range(class_num):
        # 1. 识别纯色：找到所有最高概率名称为当前 label 的颜色点的索引
        pure_colors_idx = np.array(np.where(w2cM == label))
        # 对应这些颜色点，它们被命名为当前 label 的概率
        pure_colors_prob = W2C[pure_colors_idx, label]

        # --- (此处省略了注释掉的可视化代码) ---
        # print(label, pure_colors_idx.shape)
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(4,2))
        # ax = plt.subplot(111)
        # nt, bins, patches = plt.hist(np.array(pure_colors_prob).flatten(), density=True, color='red',bins=10, range=(0,1))
        # # plt.title('Probability distribution of color '+ COLOR_NAME[label])
        # print(id_joost2label[label].name, nt)
        # plt.xlim(0, 1)
        # plt.ylim(0, 10)
        # plt.xlabel('Probability', fontsize=18)
        # plt.ylabel('Frequency', fontsize=18)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)

        # # plt.show()
        # plt.savefig('./results/'+ id_ca2label[label].name + '_joost.png', bbox_inches='tight')
        # plt.close()

        pure_colors_num = np.size(pure_colors_prob, 1)
        sort_idx = np.argsort(pure_colors_prob)
        # 2. 筛选：根据 color_percent 选取概率最高的颜色子集
        pure_num = np.round(pure_colors_num * color_percent).astype(np.int32)
        selected_idx = np.array(pure_colors_idx[0:, sort_idx[0, -pure_num:]])
        selected_prob = W2C[selected_idx, label]

        # 提取选中颜色的 RGB/HSV/Lab 值
        rgb = np.squeeze(W2CRGB[selected_idx, :], axis=0)
        hsv = np.squeeze(W2CHSV[selected_idx, :], axis=0)
        lab = np.squeeze(W2CLAB[selected_idx, :], axis=0)

        # 准备聚类样本：根据颜色空间和模式选择要聚类的维度
        if colorspace == 'lab':
            if mode == 2:
                samples = lab[:, 1:]  # 仅使用 a*b* 通道
            else:
                samples = lab  # 使用 L*a*b* 全通道
        elif colorspace == 'rgb':
            samples = rgb

        # 确定聚类数量（考虑是否为特殊颜色需要额外的原型）
        if name_type == 'joost':
            if_extra_colors = id_joost2label[label].extra_color
        # ... (对其他 name_type 的处理)
        # ...

        cluster_num_m = cluster_num
        if if_extra_colors:
            cluster_num_m = cluster_num + extra_num

        # 3. K-Means 聚类
        # 使用颜色命名概率作为权重，使得高概率的颜色点对聚类中心的影响更大
        kmeans_f_2 = KMeans(n_clusters=cluster_num_m, random_state=0, n_init=cluster_num_m).fit(
            samples, y=None, sample_weight=selected_prob.flatten())

        # 预测每个筛选颜色点属于哪个聚类
        img_labels = kmeans_f_2.predict(samples)
        top_colors_rgb = np.zeros((cluster_num_m, 3))
        top_colors_hsv = np.zeros((cluster_num_m, 3))
        top_colors_lab = np.zeros((cluster_num_m, 3))

        # 4. 原型选择 (从每个聚类中选出一个颜色作为原型)
        if cluster_type == 'saturation':
            # 基于饱和度：选择聚类中饱和度最高的颜色作为原型
            for center in range(cluster_num_m):
                center_idx = np.array(np.where(img_labels == center))
                color = hsv[center_idx, :]
                # 找到 H 通道中饱和度（第二个通道）最高的颜色
                sat_rank = np.argsort(color[:, :, 1])[:, -1]
                # 记录该颜色点的 Lab/RGB/HSV 值
                top_colors_rgb[center, :] = rgb[center_idx[0, sat_rank.flatten()], :]
                top_colors_hsv[center, :] = hsv[center_idx[0, sat_rank.flatten()], :]
                top_colors_lab[center, :] = lab[center_idx[0, sat_rank.flatten()], :]


        elif cluster_type == 'prob':
            # 基于概率：选择聚类中 W2C 概率最高的颜色作为原型
            for center in range(cluster_num_m):
                center_idx = np.array(np.where(img_labels == center))
                prob = selected_prob[:, center_idx[0]]
                # 找到聚类中概率最高的颜色
                sat_rank = np.argsort(prob)[:, -1]
                top_colors_rgb[center, :] = rgb[center_idx[0, sat_rank.flatten()], :]
                top_colors_hsv[center, :] = hsv[center_idx[0, sat_rank.flatten()], :]
                top_colors_lab[center, :] = lab[center_idx[0, sat_rank.flatten()], :]


        elif cluster_type == 'probysat':
            # 基于概率和饱和度（更精细）：
            # 1. 首先在聚类中筛选出概率最高的 `sub_color_percent` 子集。
            # 2. 然后在该子集中选择饱和度最高的颜色作为原型。
            for center in range(cluster_num_m):
                center_idx = np.array(np.where(img_labels == center))
                prob = selected_prob[:, center_idx[0]]
                prob_rank = np.argsort(prob)

                # 筛选高概率子集
                sub_pure_num = np.round(np.size(prob, 1) * sub_color_percent).astype(np.int32)
                sub_selected_idx = np.array(center_idx[0:, prob_rank[0, -sub_pure_num:]])

                # 在高概率子集中，选择饱和度最高的颜色
                color = hsv[sub_selected_idx, :]
                sat_rank = np.argsort(color[:, :, 1])[:, -1]

                top_colors_rgb[center, :] = rgb[sub_selected_idx[0, sat_rank.flatten()], :]
                top_colors_hsv[center, :] = hsv[sub_selected_idx[0, sat_rank.flatten()], :]
                top_colors_lab[center, :] = lab[sub_selected_idx[0, sat_rank.flatten()], :]

        # 将原型颜色添加到列表
        prototype_rgb.append(top_colors_rgb)
        prototype_lab.append(top_colors_lab)
        prototype_hsv.append(top_colors_hsv)
        num_prototype.append(cluster_num_m)

    return prototype_rgb, prototype_lab, prototype_hsv, num_prototype


def compare_color_name(img_src, img_tgt, W2C, name_type='joost', threshold=0.98):
    """
    比较图像的颜色命名是否一致。

    基本原理：计算源图像和目标图像在颜色命名概率分布上的差异。
    """

    color_label_org, color_nam_org, _, prob_map_org = img2color(img_src, W2C, name_type=name_type)
    color_label_new, color_nam_new, _, prob_map_new = img2color(img_tgt, W2C, name_type=name_type)

    if threshold == 0:
        # 简单比较：颜色命名标签是否完全相同
        is_same_color = (color_label_org == color_label_new)
    else:
        # 精确比较：计算概率分布的 L2 距离
        diff = np.zeros_like(color_label_org).astype(np.float64)
        for jj in range(np.size(color_label_org, 0)):
            for ii in range(np.size(color_label_org, 1)):
                # 计算两个颜色概率分布向量之间的 L2 距离
                diff[jj, ii] = np.linalg.norm(prob_map_org[jj, ii, :] - prob_map_new[jj, ii, :])
                # 判断 L2 距离是否小于阈值
        is_same_color = (np.abs(diff) < threshold)

    print(is_same_color)
    return is_same_color


def find_similar_color_topkname(img_src, prototype_rgb, prototype_lab, prototype_hsv,
                                W2C, name_type='joost',
                                dist_type='l2', top_k=3, prob_thres=0.15,
                                colorspace='rgb', mode=2, lightness=70
                                ):
    """
    基于颜色命名概率，从原型颜色中寻找与源颜色最相似的颜色。

    算法步骤：
    1. 概率筛选：计算源颜色 `img_src` 的颜色命名概率，只考虑概率最高的 top-k 个颜色名称。
    2. 子集构建：将这些 top-k 颜色名称对应的原型颜色合并成一个**目标颜色子集**。
    3. 距离计算：在指定的颜色空间（Lab, RGB, HSV）中，计算 `img_src` 与目标颜色子集中所有颜色的距离（L2, L1, 角度）。
    4. 目标确定：选择距离最小的颜色作为目标颜色 `target_color`。

    Args:
        img_src (np.ndarray): 源颜色（Lab 空间）。
        prototype_lab (list of np.ndarray): 所有原型颜色，按颜色名称分类。
        top_k (int): 考虑概率最高的 K 个颜色名称。
        prob_thres (float): 仅考虑概率高于此阈值的颜色名称。
        colorspace (str): 计算距离的颜色空间 ('lab', 'rgb', 'hsv')。
        dist_type (str): 距离类型 ('l2', 'l1', 'angle')。
        lightness (float): 在 ab 平面模式下，为目标颜色设置的 L 值。

    Returns:
        np.ndarray: 找到的目标颜色 `target_color` (Lab 空间)。
    """
    # 1. 概率筛选
    # prob_map: 源颜色到所有颜色名称的概率分布
    _, _, _, prob_map = img2color(img_src, W2C, name_type=name_type)
    target_color = np.zeros_like(img_src)

    for j in range(np.size(img_src, 0)):
        prob = prob_map[0, j]
        # 找到概率最大的 top_k 个颜色名称的索引
        top_index = heapq.nlargest(top_k, range(len(prob)), prob.take)

        prototype_topk_lab = []
        prototype_topk_rgb = []
        prototype_topk_hsv = []
        # 2. 子集构建：合并对应 top-k 颜色名称的原型颜色
        for k in top_index:
            if prob[k] > prob_thres:
                prototype_topk_lab.extend(prototype_lab[k])
                prototype_topk_rgb.extend(prototype_rgb[k])
                prototype_topk_hsv.extend(prototype_hsv[k])
        proto_color_num = len(prototype_topk_lab)

        # 3. 距离计算与 4. 目标确定
        if colorspace == 'lab':
            # 在 Lab 空间计算距离
            color = np.ones((proto_color_num, 1)) * img_src[j, :]

            if dist_type == 'l2':
                diff = (color - prototype_topk_lab) ** 2
            # ... (其他距离类型 l1, angle)

            if mode == 2:
                # 仅对 a*b* 通道计算距离
                diff = np.sum(diff[:, 1:], axis=1)
                idx = np.argmin(diff)
                target_color[j, 1:] = prototype_topk_lab[idx][1:]
                # 强制设置 L* 值
                target_color[j, 0] = lightness
            elif mode == 3:
                # 对 L*a*b* 全通道计算距离
                diff = np.sum(diff, axis=1)
                idx = np.argmin(diff)
                target_color[j, :] = prototype_topk_lab[idx]

        elif colorspace == 'rgb':
            # 在 RGB 空间计算距离
            img_src_1 = np.expand_dims(img_src, axis=0)
            img_src_rgb = lab2rgb(img_src_1) * 255
            img_src_rgb = np.squeeze(img_src_rgb)

            if dist_type == 'l2':
                # l2 距离
                color = np.ones((proto_color_num, 1)) * img_src_rgb[j]
                diff = (color - prototype_topk_rgb) ** 2
                diff = np.sqrt(np.sum(diff, axis=1))
            # ... (其他距离类型 l1, angle)

            idx = np.argmin(diff)
            # 目标颜色从原型 Lab 列表中获取
            target_color[j, :] = prototype_topk_lab[idx]

        # ... (对 hsv 空间的处理)

    return target_color


def find_similar_color_topkname_prob(img_src, prototype_lab, W2C, name_type='joost',
                                     top_k=3, lightness=70., prob_thres=0.15, mode=2):
    """
    基于概率分布的交叉熵/KL 散度，从原型颜色中寻找与源颜色最相似的颜色。

    算法步骤：
    1. 概率筛选：与 find_similar_color_topkname 相同，确定目标颜色子集。
    2. 目标颜色 L 值归一化：将目标颜色子集的 L* 值统一设置为 `lightness`。
    3. 概率分布计算：计算目标颜色子集中每个颜色的颜色命名概率分布。
    4. 相似度计算：计算源颜色概率分布与每个目标颜色概率分布之间的交叉熵（或其他散度）。
    5. 目标确定：选择交叉熵最小（分布最相似）的颜色作为目标颜色。

    Args:
        img_src (np.ndarray): 源颜色（Lab 空间）。
        prototype_lab (list of np.ndarray): 所有原型颜色，按颜色名称分类。
        top_k (int): 考虑概率最高的 K 个颜色名称。
        lightness (float): 为原型颜色设置的 L* 值。

    Returns:
        np.ndarray: 找到的目标颜色 `target_color` (Lab 空间)。
    """
    # 1. 概率筛选（与上一个函数相同）
    _, _, _, prob_map = img2color(img_src, W2C, name_type=name_type)
    target_color = np.zeros_like(img_src)

    for j in range(np.size(img_src, 0)):
        prob = prob_map[0, j]
        top_index = heapq.nlargest(top_k, range(len(prob)), prob.take)

        prototype_topk_lab = []
        for k in top_index:
            if prob[k] > prob_thres:
                prototype_topk_lab.extend(prototype_lab[k])

        proto_color_num = len(prototype_topk_lab)
        prototype_topk_lab = np.array(prototype_topk_lab)
        # 2. 目标颜色 L 值归一化
        prototype_topk_lab[:, 0] = lightness

        # 3. 概率分布计算：计算目标颜色子集的概率分布
        _, _, _, prob_map_pro = img2color(prototype_topk_lab, W2C, name_type=name_type)

        # 4. 相似度计算：计算交叉熵
        diff = np.zeros((proto_color_num, 1))
        for ii in range(proto_color_num):
            # 使用交叉熵衡量源颜色概率分布 `prob` 与原型颜色概率分布 `prob_map_pro[0, ii, :]` 的差异
            diff[ii] = cross_entropy(prob, prob_map_pro[0, ii, :])

            # 5. 目标确定：选择交叉熵最小的（分布最相似的）原型颜色
        idx = np.argmin(diff)
        target_color[j, :] = prototype_topk_lab[idx]

    return target_color


# --- 距离和散度计算函数 ---

def l2_distence(p, q):
    """计算 L2 (欧几里得) 距离。"""
    p = np.float_(p)
    q = np.float_(q)
    return np.linalg.norm(p - q)


def cross_entropy(p, q):
    """计算交叉熵 (Cross Entropy) - 用于衡量两个概率分布的差异。"""
    # 交叉熵 H(p, q) = -SUM [ p(i) * log2(q(i)) ]
    p = np.float_(p)
    q = np.float_(q)
    # 避免 log(0) 错误，通常在实际应用中会加入一个极小的 epsilon
    return -sum([p[i] * np.log2(q[i] + 1e-10) for i in range(len(p))])


def KL_divergence(p, q):
    """计算 Kullback-Leibler (KL) 散度。"""
    return scipy.stats.entropy(p, q)


def JS_divergence(p, q):
    """计算 Jensen-Shannon (JS) 散度。"""
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


def angular(p, q_all, type='rgb'):
    """
    计算颜色之间的**角度距离 (Angular Distance)**。

    基本原理：将颜色视为向量，计算它们在**归一化**后的空间中的夹角。
    在 RGB 空间中，颜色向量通常被归一化以去除亮度（Grayness）的影响，从而仅比较色度。

    Args:
        p (np.ndarray): 源颜色向量。
        q_all (np.ndarray): 目标颜色向量集合。
        type (str): 颜色空间类型 ('rgb', 'ab')，决定归一化方式。

    Returns:
        np.ndarray: 角度距离数组。
    """
    ang = np.zeros((np.size(q_all, 0), 1))

    for i in range(np.size(q_all, 0)):
        q = q_all[i, :]

        # 1. 亮度/灰度归一化 (Normalization)
        if type == 'rgb':
            # 在 RGB 空间，计算灰度值以进行归一化 (去除亮度影响)
            p_gray = p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114
            q_gray = q[0] * 0.299 + q[1] * 0.587 + q[2] * 0.114
        else:
            # 对于如 Lab 的 ab 平面，无需进行亮度归一化
            p_gray = 1.
            q_gray = 1.

        # 将颜色向量除以灰度值进行归一化
        p_norm = p / (p_gray + 1e-10)  # 加一个小量避免除以零
        q_norm = q / (q_gray + 1e-10)

        # 2. 角度计算：使用向量点积公式
        l_p = np.sqrt(p_norm.dot(p_norm))
        l_q = np.sqrt(q_norm.dot(q_norm))

        # 夹角 (arccos( (p_norm . q_norm) / (||p_norm|| * ||q_norm||) ))
        # 确保点积结果在 [-1, 1] 范围内，以防浮点误差导致 arccos 失败
        dot_product = p_norm.dot(q_norm)
        cos_theta = np.clip(dot_product / (l_p * l_q), -1.0, 1.0)

        ang[i] = np.arccos(cos_theta)

    return ang