import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches

# 假设你的数据文件路径
# file_path = 'data.txt'
file_path = "D:/aaaLab/aaagraduate/SaveVideo/source/20240429/Data6/motion.txt"

# 读取文件并提取第三列数据
data = np.loadtxt(file_path, usecols=2)

# 创建图形和轴
fig, ax = plt.subplots()
scatter = ax.scatter([], [], c=[], s=5)  # 使用'scatter'函数，s设置点的大小

# 设置轴的范围
ax.set_xlim(0, len(data))
ax.set_ylim(np.min(data), np.max(data))
ax.set_xlabel('Amplitude')
ax.set_ylabel('Frames')

# 自定义颜色映射
cmap = ListedColormap(['#1f77b4', 'orange', 'red'])
bounds = [0, 10, 15, np.max(data)+1]  # 确保边界是单调递增的并且包含所有数据
norm = BoundaryNorm(bounds, cmap.N)

danger_start = None
rectangles = []
rect_plot_times = []

# 初始化函数
def init():
    
    scatter.set_offsets(np.empty((0, 2)))
    scatter.set_array(np.empty(0))
    return scatter,

# 更新函数
def update(frame):
    global danger_start, rectangles, rect_plot_times
    # global rectangles
    xdata = np.arange(frame + 1)
    ydata = data[:frame + 1]
    scatter.set_offsets(np.c_[xdata, ydata])
    scatter.set_array(ydata)
    scatter.set_cmap(cmap)
    scatter.set_norm(norm)
    
    # print(frame)
    # if ydata[frame] > 15 and danger_start is None:
    #     danger_start = frame

    # if ydata[frame] <= 15 and danger_start is not None:
    #     rect = patches.Rectangle((danger_start, np.min(ydata)), frame - danger_start, np.max(ydata) - np.min(ydata), linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
    #     ax.add_patch(rect)
    #     rectangles.append(rect)
    #     danger_start = None
    # 检查当前数据点是否超过阈值
    # print(ydata[-1])
    if ydata[-1] > 15 and danger_start is None:
        danger_start = frame
    elif ydata[-1] <= 15 and danger_start is not None:
        print(1)
        rect = patches.Rectangle((danger_start, np.min(ydata)), frame - danger_start, np.max(ydata) - np.min(ydata), linewidth=1, edgecolor='none', facecolor='red', alpha=0.02)
        ax.add_patch(rect)
        rectangles.append(rect)
        rect_plot_times.append(1)
        danger_start = None

    # # 确保所有矩形被重新绘制
    for idx, rect in enumerate(rectangles):
        if(rect_plot_times[idx] < 20):
            ax.add_patch(rect)
            rect_plot_times[idx] += rect_plot_times[idx]

    return scatter,

# # 初始化函数
# def init():
#     scatter.set_offsets(np.empty((0, 2)))
#     scatter.set_array(np.empty(0))
#     return scatter,

# # 更新函数
# def update(frame):
#     # global rectangles
#     xdata = np.arange(frame + 1)
#     ydata = data[:frame + 1]
#     scatter.set_offsets(np.c_[xdata, ydata])
#     scatter.set_array(ydata)
#     scatter.set_cmap(cmap)
#     scatter.set_norm(norm)
    
#     # # 清除之前的矩形
#     # for rect in rectangles:
#     #     rect.remove()
#     # rectangles = []

#     # 检查蓝色区间并添加矩形
#     blue_start = None
#     for i in range(frame + 1):
#         print(i)
#         if ydata[i] > 15:
#             if blue_start is None:
#                 blue_start = i
#         else:
#             if blue_start is not None:
#                 # print(1)
#                 print((blue_start, np.min(ydata)), i - blue_start, np.max(ydata) - np.min(ydata))
#                 rect = patches.Rectangle((blue_start, np.min(ydata)), i - blue_start, np.max(ydata) - np.min(ydata), linewidth=1, edgecolor='blue', facecolor='blue', alpha=0.3)
#                 ax.add_patch(rect)
#                 # rectangles.append(rect)
#                 blue_start = None
#                 break
#     # if blue_start is not None:
#     #     rect = patches.Rectangle((blue_start, np.min(ydata)), frame + 1 - blue_start, np.max(ydata) - np.min(ydata), linewidth=1, edgecolor='none', facecolor='blue', alpha=0.3)
#     #     ax.add_patch(rect)
#     #     rectangles.append(rect)
#     return scatter,


# 设置动画速度，interval表示每帧之间的间隔时间，单位为毫秒
animation_speed = 33  # 调整此值以改变动画速度，100毫秒表示每秒10帧

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(data), init_func=init, blit=False, repeat=False, interval=animation_speed)
# 保存动画
ani.save('D:/aaaLab/aaagraduate/SaveVideo/DetectionApp/animation.gif', writer='Pillow', fps=30) 

# 显示动画
plt.show()


