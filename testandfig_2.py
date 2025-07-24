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
            continue  # 跳过帧

        delta_traj = policy.process_frame(image=frame)  # shape: (16, 14)
        delta_traj = np.array(delta_traj, dtype=np.float32)
        if delta_traj.shape != (16, 15):
            print(f"Frame {frame_idx}: invalid shape {delta_traj.shape}")
            frame_idx += 1
            continue

        delta_left_xyz = delta_traj[:, :3]
        left_gripper = delta_traj[-1, 6]
        delta_right_xyz = delta_traj[:, 7:10]
        right_gripper = delta_traj[-1, 13]
        action_arm = delta_traj[-1, 14]

        current_xyz_left += np.sum(delta_left_xyz, axis=0)
        current_xyz_right += np.sum(delta_right_xyz, axis=0)

        left_part = ' '.join(f'{x:.6f}' for x in current_xyz_left) + f' {left_gripper:.6f}'
        right_part = ' '.join(f'{x:.6f}' for x in current_xyz_right) + f' {right_gripper:.6f}'
        line = f'{left_part} || {right_part} || {action_arm:.6f}'

        f.write(line + '\n')
        print(f"Saved frame {saved_idx} (video frame {frame_idx}): {line}")
        saved_idx += 1
        frame_idx += 1

cap.release()
cv2.destroyAllWindows()
