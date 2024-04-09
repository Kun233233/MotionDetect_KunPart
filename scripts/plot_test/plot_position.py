import numpy as np
import matplotlib.pyplot as plt

import os

def plot_feature_points_position(data_file, save_path = None, time_range=None):
    # 读取数据文件
    # 打开txt文件进行读取
    with open(data_file, "r") as file:
        # 读取文件内容
        data = file.read()
    

    # 分割每一行数据
    lines = data.strip().split('\n')
    
    print(len(lines))
    time = np.arange(len(lines))
    feature_position = np.empty((len(lines), 6, 3))
    labels = ['NoseTop', 'LeftL', 'LeftR', 'RightL', 'RightR', 'MouseTop']
    titles = ['x-direction', 'y-direction', 'z-direction']

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

    # 创建子图
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # 绘制每个变量
    for axis, ax in enumerate(axs.flat):
        for feature in range(0,6):
            ax.plot(time, feature_position[:, feature, axis], label=labels[feature])
        ax.set_title(titles[axis], fontsize=16)
        ax.set_xlabel('Frame', fontsize=16)
        ax.set_ylabel('Distance(mm)', fontsize=16)
        # 设置坐标轴上刻度数字的大小
        ax.tick_params(axis='x', labelsize=14)  # 设置 x 轴刻度数字的大小为 14
        ax.tick_params(axis='y', labelsize=14)  # 设置 y 轴刻度数字的大小为 14
        ax.legend()

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

    data_file = f"{script_directory}/../../source/20240326/points3.txt"
    save_path = f"{script_directory}/../../source/20240326/points3.png"
    # plot_motion(data_file=data_file, time_range=(300, 1200))
    plot_feature_points_position(data_file=data_file, save_path=save_path)
