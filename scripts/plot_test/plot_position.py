import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os

def plot_feature_points_position(data_file, save_path = None, time_range=None):

    # labels = ['NoseTop', 'LeftL', 'LeftR', 'RightL', 'RightR', 'MouseTop']
    labels = ['Feature point 22', 'Feature point 23', 'Feature point 30', 'Feature point 31', 'Feature point 37', 'Feature point 40',
              'Feature point 43', 'Feature point 46', 'Feature point 49', 'Feature point 52', 'Feature point 55', 'Feature point 58']
    titles = ['x-direction', 'y-direction', 'z-direction']
    # 读取数据文件
    # 打开txt文件进行读取
    with open(data_file, "r") as file:
        # 读取文件内容
        data = file.read()
    

    # 分割每一行数据
    lines = data.strip().split('\n')
    
    print(len(lines))
    time = np.arange(len(lines))
    feature_position = np.empty((len(lines), len(labels), 3))
    

    # 解析每行数据
    row_now = 0
    column_now = 0
    for line in lines:
        coordinates = line.strip().split()
        for coord in coordinates:
            # 去除括号并分割成坐标值
            x, y, z = coord.strip('()').split(',')
            feature_position[row_now, column_now, 0] = float(x)
            feature_position[row_now, column_now, 1] = float(y)
            feature_position[row_now, column_now, 2] = float(z)
            # print("x:", x, "y:", y, "z:", z)
            column_now = column_now + 1
        column_now = 0
        row_now = row_now + 1


    for label in range(len(labels)):
        f_now = feature_position[:, label] # 取出label对应的每个方向的数据            
        # 过滤出非零数据的索引和值
        nonzero_mask = (f_now[:, 0] != 0) & (f_now[:, 1] != 0) & (f_now[:, 2] != -1)  # 筛选出符合条件的点
        # print(f_now[:, 2] > 590)
        # print(2)
        nonzero_indices_xyz = np.where(nonzero_mask)[0] # 找到z方向为-1的数据点，x 和 y方向为-1的点
        nonzero_values_xyz = f_now[nonzero_indices_xyz] 

        for axis in range(3):
            nonzero_indices = nonzero_indices_xyz
            nonzero_values = nonzero_values_xyz[:, axis]
            # print(nonzero_indices_xyz.shape)
            # import pdb; pdb.set_trace()
            # 创建插值函数
            interp_func = interp1d(nonzero_indices, nonzero_values, kind='cubic')
            # 生成插值后的数据
            interpolated_indices = np.arange(len(lines))
            interpolated_values = interp_func(interpolated_indices)
            feature_position[:, label, axis] = interpolated_values
            # print(feature_position[:, label, axis])
    # print(feature_position.shape)
    

    # for label in range(len(labels)):
    #     f_now = feature_position[:, label] # 取出label对应的每个方向的数据            
    #     # 过滤出非零数据的索引和值
    #     # nonzero_mask = f_now[:, 2] > 585  # 筛选出符合条件的点
    #     # print(f_now[:, 2] > 590)
    #     # print(2)
    #     nonzero_indices_xyz = np.where(f_now[:, 2] > 600)[0] # 找到z方向为-1的数据点，x 和 y方向为-1的点
    #     nonzero_values_xyz = f_now[nonzero_indices_xyz] 

    #     for axis in range(3):
    #         nonzero_indices = nonzero_indices_xyz
    #         nonzero_values = nonzero_values_xyz[:, axis]
    #         print(nonzero_indices_xyz.shape)
    #         # import pdb; pdb.set_trace()
    #         # 创建插值函数
    #         interp_func = interp1d(nonzero_indices, nonzero_values, kind='cubic')
    #         # 生成插值后的数据
    #         interpolated_indices = np.arange(len(lines))
    #         interpolated_values = interp_func(interpolated_indices)
    #         feature_position[:, label, axis] = interpolated_values


    # 创建子图
    # fig, axs = plt.subplots(3, 1, figsize=(8, 12))
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    first_subfig = True
    # 绘制每个变量
    for axis, ax in enumerate(axs.flat):
        for feature in range(0,len(labels)):
            ax.plot(time, feature_position[:, feature, axis], label=labels[feature])
        # ax.set_title(titles[axis], fontsize=16)
        ax.set_xlabel('Frame', fontsize=16)
        if first_subfig:
            ax.set_ylabel('Distance(mm)', fontsize=16)
            first_subfig = False
        # 设置坐标轴上刻度数字的大小
        ax.tick_params(axis='x', labelsize=14)  # 设置 x 轴刻度数字的大小为 14
        ax.tick_params(axis='y', labelsize=14)  # 设置 y 轴刻度数字的大小为 14
        # ax.legend()
    # fig.legend()

    # 获取所有子图中的标签和句柄
    handles, labels = [], []
    for ax in fig.axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    # 仅显示前 n 个标签
    n = len(labels)  # 设置只显示前3个标签
    # fig.legend(handles[:n], labels[:n], loc='upper center', bbox_to_anchor=(0.92, 0.5), ncol=1)
    # plt.legend(handles[:n], labels[:n], bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)
    # legend = fig.legend(mode='expand')
    # legend.set_alpha(0.0)

    

    # for axis, ax in enumerate(axs.flat):
    #     for feature in range(0, 6):
    #         # 获取非零值的索引
    #         non_zero_indices = np.where(feature_position[:, feature, axis] != 0)[0]
    #         # 获取非零值对应的时间和特征值
    #         non_zero_time = time[non_zero_indices]
    #         non_zero_feature = feature_position[:, feature, axis][non_zero_indices]
    #         # 绘制非零值的曲线
    #         ax.plot(non_zero_time, non_zero_feature, label=labels[feature])
    #     ax.set_title(titles[axis])
    #     ax.set_xlabel('Frame')
    #     ax.set_ylabel('Distance(mm)')
    #     ax.legend()

    # 调整子图之间的间距
    plt.tight_layout()

    # # 添加标题和标签
    # plt.title('Variables Over Time')
    # plt.xlabel('Time')
    # plt.ylabel('Value')

    # # 添加图例
    # plt.legend()
    if save_path is not None:
        plt.savefig(save_path)

    # 显示图形
    plt.show()
    



if __name__ == '__main__':
    # print("main")
    # 获取当前脚本文件的绝对路径
    script_path = os.path.abspath(__file__)

    # 获取当前脚本所在目录的绝对路径
    script_directory = os.path.dirname(script_path)
    script_directory = script_directory.replace('\\', '/')

    print("当前脚本文件的绝对路径:", script_path)
    print("当前脚本所在目录的绝对路径:", script_directory)
    # data_file = f"{script_directory}/../../source/motion.txt"
    # data_file = f"{script_directory}/../../source/20240314/motion1.txt"
    # 静止 上下 左右 综合
    data_file = f"{script_directory}/../../source/20240429/数据与电脑录屏/静止/point_12.txt"
    save_path = f"{script_directory}/../../source/20240429/数据与电脑录屏/静止/point_12.png"
    # plot_motion(data_file=data_file, time_range=(300, 1200))
    plot_feature_points_position(data_file=data_file, save_path=save_path)
