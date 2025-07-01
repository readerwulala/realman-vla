## test.py 测试脚本，直接运行就行
import cv2
from policy_client import PolicyClient

policy = PolicyClient(base_url='http://localhost:2345')

cap = cv2.VideoCapture('250606_101355.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 传入当前帧做推理
    action = policy.process_frame(image=frame)
    print(action)

cap.release()
cv2.destroyAllWindows()