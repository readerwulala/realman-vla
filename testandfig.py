import cv2
import numpy as np
import os
from policy_client import PolicyClient

policy = PolicyClient(base_url='http://localhost:2345')
cap = cv2.VideoCapture('250303_102830_793.mp4')

# 创建 logs 文件夹
os.makedirs('logs', exist_ok=True)
output_path = os.path.join('logs', 'trajectories.txt')

current_xyz = np.zeros(3)
frame_idx = 0

with open(output_path, 'w') as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        delta_traj = policy.process_frame(image=frame)  # shape: (16, 7)
        delta_traj = np.array(delta_traj, dtype=np.float32)
        if delta_traj.shape != (16, 7):
            print(f"Frame {frame_idx}: invalid shape {delta_traj.shape}")
            continue

        total_delta_xyz = np.sum(delta_traj[:, :3], axis=0)
        current_xyz += total_delta_xyz

        line = ','.join(f'{x:.6f}' for x in current_xyz)
        f.write(line + '\n')

        print(f"Saved frame {frame_idx}: {line}")
        frame_idx += 1

cap.release()
cv2.destroyAllWindows()
