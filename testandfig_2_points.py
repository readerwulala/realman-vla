import cv2
import numpy as np
import os
from policy_client import PolicyClient

policy = PolicyClient(base_url='http://localhost:2345')
cap = cv2.VideoCapture('right_test.mp4')

os.makedirs('logs', exist_ok=True)
output_path = os.path.join('logs', 'trajectories.txt')

current_xyz_left = np.zeros(3)
current_xyz_right = np.zeros(3)

frame_interval = 1  # 每5帧处理一帧
frame_idx = 0
saved_idx = 0

with open(output_path, 'w') as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        delta_traj = policy.process_frame(image=frame)  # shape: (16, 15)
        delta_traj = np.array(delta_traj, dtype=np.float32)
        if delta_traj.shape != (16, 15):
            print(f"Frame {frame_idx}: invalid shape {delta_traj.shape}")
            frame_idx += 1
            continue

        # 展平写入
        flattened = delta_traj.flatten()  # shape: (16*15,) = (240,)
        line = ' '.join(f'{x:.6f}' for x in flattened)
        f.write(line + '\n')

        print(f"Saved frame {saved_idx} (video frame {frame_idx})")
        saved_idx += 1
        frame_idx += 1
cap.release()
cv2.destroyAllWindows()
