import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取轨迹文件
traj = np.loadtxt('logs/trajectories.txt', delimiter=',')

# 检查维度
assert traj.shape[1] == 3, "每行应为一个 xyz 点"

# 提取坐标
x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]

# 绘图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, marker='o', linewidth=2, markersize=3, label='Trajectory')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Accumulated Trajectory')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()
