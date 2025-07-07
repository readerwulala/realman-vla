# -*- coding: utf-8 -*-
#这是一个用视频测试的脚本
import cv2
import requests
import tempfile
import os
import json
from tqdm import tqdm
import subprocess

video_path = "250606_101355.mp4"
url = "http://127.0.0.1:2345/process_frame"
text_prompt = "Pick up the green straw and insert it into the black cup"
output_file = "trajectory_results.jl"  # 改成 .jl（JSON Lines）

def split_and_reinfer(video_path):
    """
    这个函数示例：把视频按 X 秒一段切片，
    然后对每个切片里的关键帧重新上传推理。
    你可以根据实际需求改这里的逻辑。
    """
    # 假设每 5 秒切一段
    segment_dir = "segments"
    os.makedirs(segment_dir, exist_ok=True)
    # ffmpeg 切片
    subprocess.run([
        "ffmpeg", "-i", video_path, "-c", "copy",
        "-map", "0", "-segment_time", "5", "-f", "segment",
        os.path.join(segment_dir, "seg_%03d.mp4")
    ], check=True)
    # 对每个切片再做推理（这里只打印名字，替换你自己的逻辑）
    for seg in sorted(os.listdir(segment_dir)):
        path = os.path.join(segment_dir, seg)
        print(f"Re-infer on segment: {path}")
        # ……上传/请求代码同上……

def main():
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 如果文件已存在，删掉
    if os.path.exists(output_file):
        os.remove(output_file)

    with tqdm(total=frame_count, desc="推理进度") as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 临时存图
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                tmp_path = tmp.name

            # 上传请求
            print(f"[Frame {frame_idx}] 上传图片中...")  # 上传提示
            with open(tmp_path, "rb") as img_f:
                files = [("image", ("frame.jpg", img_f, "image/jpeg"))]
                data = {"text": text_prompt}
                resp = requests.post(url, data=data, files=files)

            os.remove(tmp_path)

            # 解析并**追加**写入一行
            if resp.ok:
                delta_actions = resp.json()["response"]
                print(type(delta_actions))
            else:
                print(f"[Frame {frame_idx}] 请求失败:", resp.status_code)
                delta_actions = []
            ###
            for action_idx, single_action in enumerate(delta_actions, 1):
                delta_left_actions, left_gripper = single_action[:-1], single_action[-1]
                print(f"Action {action_idx}: Delta_pose: {delta_left_actions}, Gripper: {left_gripper}")

                row_data = {
                    "frame": frame_idx,
                    "step": action_idx,
                    "trajectory": single_action
                }

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row_data, ensure_ascii=False) + "\n")

                print(f"[Frame {frame_idx} - Step {action_idx}] 写入成功：{row_data}")
            pbar.update(1)
            frame_idx += 1


    cap.release()

    print("所有帧推理完毕，已逐行写入到", output_file)
    # 再去切视频并做新一轮推理
    split_and_reinfer(video_path)

if __name__ == "__main__":
    main()
