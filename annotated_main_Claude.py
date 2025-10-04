"""
基于颜色命名的调色板颜色协调化算法
==================================

核心思想：
该算法通过颜色命名模型将颜色空间分为11个基本颜色类别（红、橙、棕、黄、绿、蓝、紫、粉、黑、灰、白），
然后通过将图像的调色板映射到具有相同颜色名称但饱和度更高的原型颜色上，实现颜色协调化。

主要步骤：
1. 原型调色板生成：通过颜色命名和聚类生成高饱和度的候选颜色
2. 调色板提取：从输入图像中提取代表性颜色
3. 颜色匹配：将提取的颜色匹配到相同颜色名称的原型颜色
4. 图像重新着色：基于匹配的调色板对图像进行重新着色
"""

import os
import logging
import argparse
import cv2
import time
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2hsv, hsv2rgb # 导入颜色空间转换函数
from extract_palette import histogram, extract_palette # 导入调色板提取相关函数，用于从图像中提取代表性颜色
from recolor_Claude import lab_transfer, rgb_transfer  # 导入图像重着色函数，用于基于调色板映射修改图像颜色
from color_naming.color_name_compare_Gemini import color_clustering, find_similar_color_topkname, \
    find_similar_color_topkname_prob  # 导入颜色命名和比较模块，用于生成原型调色板和匹配颜色
from color_naming.color_naming import load_colornamelut  # 导入颜色命名查找表（LUT）加载函数，用于颜色分类
from utils import visualize_palette_rgb, draw_comparison  # 导入可视化工具函数，用于绘制调色板和比较图像
from eval import evaluation  # 导入评估函数，用于计算图像质量指标和和谐度

def palette_color_harmony(args):
    """
    主函数：执行基于调色板的颜色协调化
    
    算法流程：
    1. 首先生成与图像无关的原型调色板（所有图像共享）
    2. 对每张图像：
       a. 提取图像的颜色调色板
       b. 将调色板颜色匹配到原型颜色
       c. 使用新调色板重新着色图像
    """
    
    # ========== 准备工作：加载图像列表 ==========
    images = os.listdir(args.data_dir)
    # 筛选支持的图像格式
    images = [name for name in images if name.endswith('.jpg') or name.endswith('.jpeg') or name.endswith('.png')]

    num_img = len(images)

    if num_img == 0:
        logging.info('There are no images at directory %s. Check the data path.' % args.data_dir)
    else:
        logging.info('There are %d images to be processed.' % num_img)
    images.sort()

    # 构建实验标签，用于保存结果时的文件命名
    # 标签包含了所有关键参数，便于追踪不同实验设置
    exp_label = "RE_palette_lab{}_cluster_{}{}_c{}_mapping_{}{}_recolor_{}{}".format(
                args.palette_mode,      # 调色板提取模式（2:ab通道, 3:lab通道）
                args.cluster_space,      # 聚类颜色空间（rgb/lab/hsv）
                args.cluster_mode,       # 聚类通道数
                args.cluster_num,        # 每个颜色名称的聚类中心数
                args.mapping_space,      # 匹配时使用的颜色空间
                args.mapping_mode,       # 匹配时使用的通道
                args.recolor_space,      # 重新着色的颜色空间
                args.recolor_mode)       # 重新着色的模式

    save_dir = os.path.join(args.save_dir, exp_label)

    # 创建保存目录结构
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'palette'))   # 保存调色板对比图
        os.makedirs(os.path.join(save_dir, 'image'))     # 保存处理后的图像
        os.makedirs(os.path.join(save_dir, 'harmony'))   # 保存和谐度分析图

    # 创建CSV文件记录评估指标
    with open(f'{save_dir}/{exp_label}.csv', 'w') as f:
        f.write('image,time,niqe,brisque,best_temp,best_alpha,Fscore,score_sat,score_perct,width,height\n')

    # ==================== 步骤1：生成原型调色板 ====================
    # 这是算法的核心创新点：通过颜色命名模型生成高饱和度的原型颜色
    # 这些原型颜色将作为目标颜色，用于后续的颜色映射
    
    # 加载颜色命名模型（将RGB值映射到11个基本颜色名称的概率分布）
    W2C = load_colornamelut(name_type=args.name_method)
    
    T1 = time.time()
    
    # 执行颜色聚类，生成原型调色板
    # 原理：
    # 1. 对整个RGB颜色空间进行采样（8x8x8网格）
    # 2. 使用颜色命名模型将每个颜色分配到11个类别之一
    # 3. 对每个颜色类别进行K-means聚类
    # 4. 选择每个聚类中饱和度最高的颜色作为原型颜色
    prototype_rgb, prototype_lab, prototype_hsv, num_proto = color_clustering(
        args.cluster_num,           # 每个颜色名称的基础聚类数
        args.cluster_prob,          # 概率阈值，筛选高置信度的颜色
        W2C,                        # 颜色命名查找表
        name_type=args.name_method, # 颜色命名方法（joost/ca/yu）
        extra_num=args.extra_num,   # 为常见颜色（绿、蓝、紫、红、粉）增加的额外聚类数
        cluster_type=args.cluster_type,  # 选择原型的策略（saturation/prob/probysat）
        colorspace=args.cluster_space,   # 聚类使用的颜色空间
        mode=args.cluster_mode           # 使用哪些通道进行聚类
    )
    
    T2 = time.time()
    time_single = T2-T1
    print('Time for prototype palettes generation: ', time_single)

    # 可视化并保存每个颜色类别的原型调色板
    for label, proto in enumerate(prototype_rgb):
        palette = visualize_palette_rgb(proto, patch_size=20)
        palette = np.array(palette).astype(np.uint8)
        out_img_path = os.path.join(save_dir, 'palette_cluster_label{}.jpg'.format(label))
        cv2.imwrite(out_img_path, cv2.cvtColor(palette, cv2.COLOR_RGB2BGR))

    # 初始化评估指标累加器
    num, niqe_all, brisque_all, Fscore_all,score_sat_all,score_perct_all, time_all= 0, 0, 0, 0, 0, 0, 0

    # ========== 对每张图像进行处理 ==========
    for img_id, img_name in enumerate(images):

        img_dir = os.path.join(args.data_dir, img_name)
        img = cv2.imread(img_dir)
        print('processing image {}: {}...'.format(img_id, img_dir))

        # 调整图像大小到标准尺寸（短边512像素）
        # 这样做是为了：1.加速处理 2.保持结果一致性 3.减少内存消耗
        anchor = 512
        width = img.shape[1]
        height = img.shape[0]			
        if width >= height:
            dim = (np.floor(width/height*anchor).astype(int), anchor)
        else:
            dim = (anchor, np.floor(height/width*anchor).astype(int))
        img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

        # 颜色空间转换：BGR -> RGB -> Lab
        # Lab颜色空间的优势：L通道（亮度）与ab通道（颜色）分离，
        # 便于独立调整颜色而不影响亮度
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_lab = rgb2lab(img_rgb)
        w,h,c = img_rgb.shape

        # 创建二值掩码（这里全部为1，表示处理整个图像）
        # 可以修改为语义分割结果，实现选择性颜色调整
        label_binary = np.ones_like(img_rgb[:,:,0])

        T1 = time.time()

        # ==================== 步骤2：提取图像调色板 ====================
        # 目标：找出能够代表图像主要颜色的调色板
        
        # 计算颜色直方图
        # 原理：将Lab颜色空间划分为bin_size³个小格子，
        # 统计每个格子中的像素数量，得到颜色分布
        hist_samples, hist_counts = histogram(
            img_lab,                    # 输入图像（Lab空间）
            args.bin_size,              # 直方图的分辨率（格子数）
            mode=args.palette_mode,     # 模式2:只考虑ab通道，模式3:考虑lab所有通道
            mask=label_binary           # 掩码，指定要处理的区域
        )
        
        # 基于直方图提取调色板
        # 算法流程：
        # 1. 使用K-means聚类对直方图进行聚类
        # 2. 自动确定最优聚类数（基于解释方差百分比）
        # 3. 返回聚类中心作为调色板颜色
        c_center = extract_palette(
            img_lab,                            # 原始图像
            hist_samples,                       # 直方图采样点
            hist_counts,                        # 每个采样点的计数
            mode=args.palette_mode,             # 提取模式
            lightness=args.lightness,           # 固定亮度值（用于2D模式）
            threshold=args.palette_distortion_thres,  # 确定聚类数的阈值
            max_cluster=args.palette_num,       # 最大聚类数
            mask=label_binary                   # 处理区域掩码
        )
        
        # ==================== 步骤3：调色板匹配 ====================
        # 核心创新：保持颜色名称不变，但提高饱和度
        # 这确保了颜色的语义含义（如"天空蓝"仍然是蓝色）保持不变
        
        if args.color_dist == 'l1' or args.color_dist == 'l2' or args.color_dist == 'angle':
            # 基于距离的匹配方法
            # 工作原理：
            # 1. 对源调色板中的每个颜色，使用颜色命名模型确定其颜色名称
            # 2. 在具有相同颜色名称的原型颜色中，找到最接近的颜色
            # 3. 支持模糊匹配：如果一个颜色有多个可能的名称，都会被考虑
            c_target = find_similar_color_topkname(
                c_center,                    # 源调色板（从图像提取的）
                prototype_rgb,               # RGB空间的原型颜色
                prototype_lab,               # Lab空间的原型颜色
                prototype_hsv,               # HSV空间的原型颜色
                W2C,                         # 颜色命名模型
                name_type=args.name_method,  # 使用的颜色命名方法
                dist_type=args.color_dist,   # 距离度量类型（L1/L2/角度）
                top_k=args.topk_names,       # 考虑前k个最可能的颜色名称
                prob_thres=args.prob_thres,  # 概率阈值，过滤低置信度的颜色名称
                colorspace=args.mapping_space,  # 进行匹配的颜色空间
                lightness=args.lightness,    # 固定亮度值
                mode=args.mapping_mode       # 匹配模式（使用哪些通道）
            )

        elif args.color_dist == 'prob':
            # 基于概率的匹配方法
            # 使用颜色名称概率分布的相似度进行匹配
            c_target = find_similar_color_topkname_prob(
                c_center,                    # 源调色板
                prototype_lab,               # Lab空间的原型颜色
                W2C,                         # 颜色命名模型
                name_type=args.name_method,
                top_k=args.topk_names,
                lightness=args.lightness,
                mode=args.mapping_mode
            )
        
        elif args.color_dist == 'sat':
            # 直接增加饱和度的简单基线方法
            # 将颜色转换到HSV空间，直接将S通道乘以2
            c_center_rgb = lab2rgb(np.expand_dims(c_center, axis=0))
            c_center_hsv = rgb2hsv(c_center_rgb)
            c_center_hsv[:,:,1] = c_center_hsv[:,:,1]*2  # 饱和度加倍
            c_target_hsv = np.minimum(c_center_hsv, 1)    # 限制在[0,1]范围内
            c_target_rgb = hsv2rgb(c_target_hsv)
            c_target = np.squeeze(rgb2lab(c_target_rgb), axis=0)

        # ==================== 步骤4：图像重新着色 ====================
        # 使用新的调色板对图像进行重新着色
        # 原理：基于反距离加权插值，将每个像素的颜色向最近的调色板颜色移动
        
        if args.recolor_space == 'lab':
            # 在Lab颜色空间进行重新着色（推荐）
            # 优势：可以独立控制亮度和颜色，避免不自然的亮度变化
            img_rgb_naming, _ = lab_transfer(
                img_lab,                # 输入图像（Lab空间）
                c_center,               # 源调色板
                c_target,               # 目标调色板
                mask=label_binary,      # 处理区域掩码
                mode=args.recolor_mode  # 模式2:只改变ab通道，模式3:改变所有通道
            )
        else:
            # 在RGB颜色空间进行重新着色
            img_rgb_naming = rgb_transfer(
                img_rgb,                # 输入图像（RGB空间）
                c_center,               # 源调色板
                c_target,               # 目标调色板
                mask=label_binary       # 处理区域掩码
            )
        
        T2 = time.time()
        time_single = T2-T1
        print('Time for image processing: ', time_single)

        # ==================== 评估和可视化 ====================
        
        # 保存重新着色的图像
        img_rgb_naming = np.array(img_rgb_naming*255).astype(np.uint8)
        out_img_path = os.path.join(save_dir, 'image', img_name.split('/')[-1][:-4]+'.png')
        cv2.imwrite(out_img_path, cv2.cvtColor(img_rgb_naming, cv2.COLOR_RGB2BGR))

        # 如果启用评估，计算各项质量指标
        if args.eval:
            # 评估指标包括：
            # - NIQE/BRISQUE: 图像质量评估
            # - F-score: 颜色和谐度评分
            # - MITS: 模板内平均饱和度
            # - PIT: 模板内像素百分比
            niqe, brisque, temp, alpha, Fscore, score_sat, score_perct, canvas, overlay = evaluation(
                cv2.cvtColor(img_rgb_naming, cv2.COLOR_RGB2BGR)
            )

            # 记录评估结果到CSV文件
            with open(f'{save_dir}/{exp_label}.csv', 'a') as f:
                f.write(f'{img_name}, {time_single}, {niqe}, {brisque}, {temp}, {alpha}, {Fscore}, {score_sat}, {score_perct},{w},{h}\n')

            # 累加指标用于计算平均值
            niqe_all = niqe_all + niqe
            brisque_all = brisque_all + brisque
            Fscore_all = Fscore_all + Fscore
            score_sat_all = score_sat_all + score_sat
            score_perct_all = score_perct_all + score_perct
            time_all = time_all + time_single
            num = num + 1

        # 保存和谐度分析图（色轮上的颜色分布）
        out_img_path = os.path.join(save_dir, 'harmony', img_name.split('/')[-1][:-4]+'.jpg')
        cv2.addWeighted(overlay, 0.5, canvas, 1 - 0.5, 0, canvas)
        cv2.imwrite(out_img_path, canvas)

        # 保存调色板对比图（显示处理前后的调色板和颜色分布）
        palette_path = os.path.join(save_dir, 'palette', img_name.split('/')[-1][:-4]+'.jpg')
        draw_comparison(c_center, c_target, img_rgb, img_rgb_naming, palette_path)

    # ========== 计算并保存平均指标 ==========
    niqe_all = niqe_all / num
    brisque_all = brisque_all / num
    Fscore_all = Fscore_all / num
    score_sat_all = score_sat_all / num
    score_perct_all = score_perct_all / num
    time_all = time_all / num

    # 将平均值写入CSV文件最后一行
    with open(f'{save_dir}/{exp_label}.csv', 'a') as f:
        f.write(f'average, {time_all}, {niqe_all}, {brisque_all}, None, None, {Fscore_all}, {score_sat_all}, {score_perct_all},{w},{h}\n')


if __name__ == '__main__':
    """
    命令行参数解析
    
    参数分为几个类别：
    1. 基础设置：数据路径、保存路径、评估开关
    2. 颜色命名相关：命名方法、额外颜色数、概率阈值等
    3. 原型生成相关：聚类数、聚类空间、聚类模式等
    4. 调色板提取相关：bin大小、调色板数量、失真阈值等
    5. 颜色匹配相关：距离度量、匹配空间、匹配模式等
    6. 重新着色相关：着色空间、着色模式、亮度值等
    """
    
    parser = argparse.ArgumentParser(description='demo')
    
    # ========== 基础设置 ==========
    parser.add_argument('--data_dir', type=str, default='./images/',
                        help='包含待处理图像的文件夹路径')
    parser.add_argument('--save_dir', type=str, default='./results/',
                        help='保存结果的路径')
    parser.add_argument('--eval', type=bool, default=True,
                        help='是否执行质量评估（会增加处理时间）')
    
    # ========== 颜色命名相关参数 ==========
    parser.add_argument('--name_method', type=str, default='joost',
                        help='颜色命名方法，可选：[joost, ca, yu]，不同方法在颜色边界定义上略有差异')
    parser.add_argument('--extra_num', type=int, default=5,
                        help='为常见颜色（绿、蓝、紫、红、粉）增加的额外原型数量，确保这些颜色有更多选择')
    parser.add_argument('--prob_thres', type=float, default=0.15,
                        help='颜色名称概率阈值，低于此值的颜色名称将被忽略')
    parser.add_argument('--topk_names', type=int, default=3,
                        help='考虑前k个最可能的颜色名称，实现模糊匹配')
    parser.add_argument('--color_dist', type=str, default='l2',
                        help='颜色差异度量方式：l1(曼哈顿距离)、l2(欧氏距离)、angle(角度)、prob(概率)、sat(直接增加饱和度)')
    
    # ========== 原型调色板生成参数 ==========
    parser.add_argument('--cluster_num', type=int, default=10,
                        help='每个颜色名称的基础聚类中心数，决定原型颜色的多样性')
    parser.add_argument('--cluster_space', type=str, default='rgb',
                        help='执行聚类的颜色空间：rgb、lab或hsv')
    parser.add_argument('--cluster_mode', type=int, default=3,
                        help='聚类使用的通道数，3表示使用所有通道，2表示只使用色度通道')
    parser.add_argument('--cluster_prob', type=float, default=1.,
                        help='选择概率高于此值的颜色进行聚类')
    parser.add_argument('--cluster_type', type=str, default='saturation',
                        help='选择原型颜色的策略：saturation(最高饱和度)、prob(最高概率)、probysat(概率×饱和度)')
    
    # ========== 颜色匹配参数 ==========
    parser.add_argument('--mapping_space', type=str, default='rgb',
                        help='执行颜色匹配的颜色空间')
    parser.add_argument('--mapping_mode', type=int, default=3,
                        help='匹配时使用的通道，3表示所有通道，2表示只使用色度通道')
    
    # ========== 调色板提取参数 ==========
    parser.add_argument('--palette_mode', type=int, default=2,
                        help='调色板提取模式：2(基于ab通道)或3(基于lab通道)')
    parser.add_argument('--bin_size', type=int, default=16,
                        help='直方图的分辨率，越大越精细但计算量越大')
    parser.add_argument('--palette_num', type=int, default=7,
                        help='最大调色板颜色数')
    parser.add_argument('--palette_distortion_thres', type=float, default=0.93,
                        help='自动确定调色板大小的失真阈值，越高调色板越小')
    
    # ========== 图像重新着色参数 ==========
    parser.add_argument('--recolor_space', type=str, default='lab',
                        help='执行重新着色的颜色空间：lab(推荐)或rgb')
    parser.add_argument('--recolor_mode', type=int, default=3,
                        help='重新着色的通道：3(所有通道)或2(只改变颜色不改变亮度)')
    parser.add_argument('--lightness', type=float, default=70.,
                        help='当使用2D模式时的固定亮度值（Lab空间的L值，范围0-100）')
    
    args = parser.parse_args()

    # 执行主函数
    palette_color_harmony(args)