import numpy as np

# 读取文件中的数字并存储在一个列表中
def read_numbers_from_file(filename):
    numbers = []
    with open(filename, 'r') as file:
        for line in file: 
            numbers.append(float(line.strip()))
    return numbers

# 计算均值、方差和标准差
def calculate_mean_variance_and_std(numbers):
    # mean = np.mean(numbers)
    # variance = np.var(numbers)
    # std = np.std(numbers)
    mean = np.mean(numbers[20:]) #前二十次不取
    variance = np.var(numbers[20:])
    std = np.std(numbers[20:])
    return mean, variance, std

# 主函数
def main():
    root = "D:/aaaLab/aaagraduate/SaveVideo/source/time/"
    # filename = root + "registration_existingimg_time_C.txt"
    # filename = root + "registration_existingimg_time_Py.txt"
    # filename = root + "get_feature_points_3D_time_C.txt"
    # filename = root + "get_feature_points_3D_time_Py.txt"
    # filename = root + "get_pose_6p_time_C.txt"
    # filename = root + "get_pose_6p_time_Py.txt"
    # filename = root + "read_mode_time_C.txt"
    # filename = root + "read_mode_time_Py.txt"
    filename = root + "read_img_time_C.txt"
    # filename = root + "read_img_time_Py.txt"

    # filename = "D:/aaaLab/aaagraduate/SaveVideo/source/time/registration_existingimg_time_C.txt"  # 文件名
    numbers = read_numbers_from_file(filename)  # 读取文件中的数字
    mean, variance, std = calculate_mean_variance_and_std(numbers)  # 计算均值、方差和标准差
    print("Mean:", mean)
    print("Variance:", variance)
    print("Standard Deviation:", std)

if __name__ == "__main__":
    main()
