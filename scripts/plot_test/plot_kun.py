import numpy as np
import matplotlib.pyplot as plt

import os

def plot_motion(data_file, save_path = None, time_range=None):
    # 读取数据文件
    # data_file = 'your_data_file.txt'
    data = np.loadtxt(data_file)

    # 提取每一列数据
    if time_range == None:
        time = np.arange(data.shape[0])  # 假设时间是递增的
        variables = [data[:, i] for i in range(6)]
    else:
        variables = [data[time_range[0]:time_range[1], i] for i in range(6)]
        time = np.arange(time_range[1] - time_range[0])


    title_var = ['x', 'y', 'z', 'a', 'b', 'c']
    # x = data[:, 0]
    # y = data[:, 1]
    # z = data[:, 2]
    # a = data[:, 2]
    # b = data[:, 2]
    # c = data[:, 2]

    # 创建子图
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # 绘制每个变量
    for i, ax in enumerate(axs.flat):
        ax.plot(time, variables[i])
        ax.set_title(title_var[i])
        ax.set_xlabel('Frame')
        if i == 0 or i == 1 or i ==2:
            ax.set_ylabel('Distance(mm)')
        else:
            ax.set_ylabel('Degree(rad)')

        ax.legend([title_var[i]])

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

    data_file = f"{script_directory}/../../source/20240326/motion3.txt"
    save_path = f"{script_directory}/../../source/20240326/motion3.png"
    # plot_motion(data_file=data_file, time_range=(300, 1200))
    plot_motion(data_file=data_file, save_path=save_path)
