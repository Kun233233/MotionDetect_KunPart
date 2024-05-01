import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def process(file_path):
    data_points = []

    with open(file_path, "r") as file:
        for line in file:
            point = [tuple(map(float, coord.strip('()').split(','))) for coord in line.strip().split()]
            data_points.append(point)

    array_3d = np.array(data_points)

    for row in range(array_3d.shape[0]):
        if np.all(np.all(array_3d[row, :, :] == array_3d[row, 0, :], axis=1)):
            array_3d[row, :, :] = [0.0, 0.0, -1.0]

    for row in range(array_3d.shape[0]):
        for col in range(array_3d.shape[1]):
            if np.array_equal(array_3d[row, col, :], [0.0, 0.0, -1.0]):
                up_index = row - 1
                while up_index >= 0:
                    if not np.array_equal(array_3d[up_index, col, :], [0.0, 0.0, -1.0]):
                        break
                    up_index -= 1
                down_index = row + 1
                while down_index < array_3d.shape[0]:
                    if not np.array_equal(array_3d[down_index, col, :], [0.0, 0.0, -1.0]):
                        break
                    down_index += 1

                if up_index >= 0 and down_index < array_3d.shape[0]:
                    up_value = array_3d[up_index, col, :]
                    down_value = array_3d[down_index, col, :]
                    distance = down_index - up_index
                    weight_up = (down_index - row) / distance
                    weight_down = (row - up_index) / distance
                    interpolated_value = weight_up * up_value + weight_down * down_value
                    array_3d[row, col, :] = interpolated_value

    n = len(array_3d)
    matrices = []
    for _ in range(n):
        matrix = np.zeros((3, 3))
        matrices.append(matrix)

    array_of_matrices = np.array(matrices)

    for i in range(n):
        array_of_matrices[i][1] = array_3d[i][0]+array_3d[i][1]+array_3d[i][5]+array_3d[i][6]-2*(array_3d[i][9]+array_3d[i][11])
        array_of_matrices[i][2] = np.cross(array_3d[i][0]+array_3d[i][5]-array_3d[i][9]-array_3d[i][11],array_3d[i][1]+array_3d[i][6]-array_3d[i][9]-array_3d[i][11])
    for i in range(n):
        array_of_matrices[i][1] = array_of_matrices[i][1]/np.linalg.norm(array_of_matrices[i][1])
        array_of_matrices[i][2] = array_of_matrices[i][2]/np.linalg.norm(array_of_matrices[i][2])
        array_of_matrices[i][0] = np.cross(array_of_matrices[i][1],array_of_matrices[i][2])

    transposed_matrices = np.array([np.transpose(matrix) for matrix in array_of_matrices])

    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    for i in range(n):
        a[i]=transposed_matrices[i][2][1]/transposed_matrices[i][2][2]
        b[i]=-transposed_matrices[i][2][0]
        c[i]=transposed_matrices[i][1][0]/transposed_matrices[i][0][0]
    for i in range(n):
        a[i]=np.degrees(np.arctan(a[i]))
        b[i]=np.degrees(np.arcsin(b[i]))
        c[i]=np.degrees(np.arctan(c[i]))
    a_ini = np.mean(a[:15])
    b_ini = np.mean(b[:15])
    c_ini = np.mean(c[:15])
    for i in range(n):
        a[i]=a[i]-a_ini
        b[i]=b[i]-b_ini
        c[i]=c[i]-c_ini

    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    for i in range(n):
        x[i]=(array_3d[i][2][0]+array_3d[i][3][0])/2+80*transposed_matrices[i][0][2]
        y[i]=(array_3d[i][2][1]+array_3d[i][3][1])/2+80*transposed_matrices[i][1][2]
        z[i]=(array_3d[i][2][2]+array_3d[i][3][2])/2+80*transposed_matrices[i][2][2]
    x_ini = np.mean(x[:15])
    y_ini = np.mean(y[:15])
    z_ini = np.mean(z[:15])
    for i in range(n):
        x[i]=x[i]-x_ini
        y[i]=y[i]-y_ini
        z[i]=z[i]-z_ini

    rotation = np.zeros(n)
    motion = np.zeros(n)
    amplitude = np.zeros(n)
    for i in range(n):
        rotation[i]=np.sqrt(np.square(a[i])+np.square(b[i])+np.square(c[i]))
        motion[i]=np.sqrt(np.square(x[i])+np.square(y[i])+np.square(z[i]))
        amplitude[i] = 0.9*rotation[i]+0.1*motion[i]

    e, d = signal.butter(8, 0.1, 'lowpass')
    a = signal.filtfilt(e, d, a,axis=0)
    b = signal.filtfilt(e, d, b,axis=0)
    c = signal.filtfilt(e, d, c,axis=0)
    x = signal.filtfilt(e, d, x,axis=0)
    y = signal.filtfilt(e, d, y,axis=0)
    z = signal.filtfilt(e, d, z,axis=0)
    rotation = signal.filtfilt(e, d, rotation,axis=0)
    motion = signal.filtfilt(e, d, motion,axis=0)
    amplitude = signal.filtfilt(e, d, amplitude,axis=0)

    colors = ['#1f77b4' if x <= 32 else 'orange' if x <= 35 else 'red' for x in amplitude]

    plt.figure('合成角位移及平动位移')
    plt.subplot(211)
    plt.plot(rotation)
    plt.xlabel('Frame')
    plt.ylabel('Rotation(°)')
    plt.title('Synthetic Rotation')
    plt.subplot(212)
    plt.plot(motion)
    plt.xlabel('Frame')
    plt.ylabel('Displacement(mm)')
    plt.title('Synthetic Displacement')
    plt.tight_layout()

    plt.figure('合成总位移')
    for i in range(n):
        plt.plot(i, amplitude[i], color=colors[i], marker='o', markersize=1.5)
    plt.xlabel('Frame')
    plt.ylabel('Amplitude')
    plt.title('Synthetic Motion')
    plt.show()
    # return plt.figure('合成角位移及平动位移'), plt.figure('合成总位移'), amplitude