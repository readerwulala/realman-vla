# -*- coding: utf-8 -*-
import threading
from traceback import print_tb
from typing import final
import numpy as np
from enum import Enum
from collections import deque
import os
import datetime
import json
# realsense
import pyrealsense2 as rs
# joy
import pygame
import sys
import cv2
import time
from ik_rbtdef import *
from ik_rbtutils import *
from policy_client import PolicyClient
from robotic_arm_package.robotic_arm import *
from ik_qp import *


class JOY(Enum):
    START_SERVER = 3
    STOP_SERVER = 0
    RESET = 7
    EXIT = 6

class StateQueue:

    def __init__(self, maxlen=10) -> None:
        self.queue = deque(maxlen=maxlen)
        self.count = 0
        self.maxlen = maxlen
    
    def enqueue(self, value):
        # value list with total length of 7
        try:
            assert len(value) == 6, "wrong with the length of pose list"
            self.queue.append(value)
            self.count = len(self.queue)
        except:
            pass

    def variance(self):
        if self.count < 2:
            return 0
        arr = np.array(list(self.queue))
        var = arr.var(axis=0)
        return list(var)


# global var
START = False #控制是否开始执行策略推理（通过手柄 START 键置为 True，STOP 键设为 False）
JOY_EVENT_RUNNING = True
MANIPLATION_RUNNIG = True #控制操作线程（机械臂+策略）是否继续执行
REALSENSE_IMAGE = None

RESET_SIGNAL = False #由手柄触发，用于重置机械臂姿态到初始位姿

# RIGTH_INIT_JOINT = [-112.0, -54.2, -15.2, -85.60, -14.7, 72.2, -33.3]
LEFT_INIT_JOINT = [ 
    -81.89600372314453,
    -63.111000061035156,
    44.849998474121094,
    -88.18399810791016,
    -3.802000045776367,
    90.06800079345703,
    9.515999794006348,
        ]
RIGHT_INIT_JOINT = [
    -111.927001953125,
    -55.15399932861328,
    -15.232999801635742,
    -85.59700012207031,
    -14.73799991607666,
    72.1520004272461,
    -33.28499984741211,
]

def init_joy():
    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("没有检测到手柄。")
        sys.exit()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"连接的手柄: {joystick.get_name()}")
    return joystick


def get_video():
    global REALSENSE_IMAGE
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # 丢弃前 30 帧
    for _ in range(30):
        if not pipeline.wait_for_frames().get_color_frame():
            continue

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
            REALSENSE_IMAGE = img
            # 本地显示，调试看效果
            #cv2.imshow("Realsense Live", img)
    finally:
        pipeline.stop()

def wait_for_realsense(threshold=100, interval=0.1):
    """
    等待 REALSENSE_IMAGE 拍到第一帧非黑屏图像。
    threshold: 均值阈值，小于此值认为是黑屏。
    interval: 轮询间隔（秒）。
    """
    print("等待摄像头图像准备中…")
    while REALSENSE_IMAGE is None or np.mean(REALSENSE_IMAGE) < threshold:
        time.sleep(interval)
    #cv2.imshow(REALSENSE_IMAGE)
    print("摄像头已就绪，启动策略线程")

def maniplation(policy_url="http://localhost:2345", right_arm_url="192.168.10.19", left_arm_url="192.168.10.18", dT=0.1):

    global RESET_SIGNAL, REALSENSE_IMAGE, MANIPLATION_RUNNIG, START

    policy = PolicyClient(base_url=policy_url)
    #print(f"{policy_url} - policy connection state: {policy.reset()}")

    # 实例化逆解库             
    qp = QPIK("RM75B", dT)
    
    #qp.set_install_angle([90, 180, 0], 'deg')
    qp.set_install_angle([0, 90, 0], 'deg')

    qp.set_work_cs_params([0, 0, 0, 0, 1.570, 0])
    qp.set_tool_cs_params([0, 0, 0, 0, 0, 0])

    #qp.set_joint_limit_max([ 178,  130,  178,  135,  178,  128, 360], 'deg')
    #qp.set_joint_limit_min([-178, -130, -178, -135, -178, -128, -360], 'deg')
    
    qp.set_joint_limit_max([ -40,  -10,  178,  135,  178,  128, 360], 'deg') #插吸管换这个
    qp.set_joint_limit_min([-140, -100, -178, -135, -178, -128, -360], 'deg')
    
    qp.set_7dof_elbow_min_angle(-135, 'deg')
    qp.set_7dof_elbow_max_angle(-3, 'deg')

    qp.set_7dof_q3_min_angle(-178,'deg')
    qp.set_7dof_q3_max_angle(178,'deg')

    # 设置运行过程中的关节速度约束
    qp.set_dq_max_weight([1, 1, 1, 1, 1, 1, 1])
    qp.set_error_weight([1, 1, 1, 1, 1, 1])

    left_arm = Arm(RM75, left_arm_url)
    left_arm.Set_Gripper_Release(500, block=False)
    #left_arm.Movej_Cmd(LEFT_INIT_JOINT, 20, 0, 0, True)
    
    right_arm = Arm(RM75, right_arm_url)
    right_arm.Set_Gripper_Release(500, block=False)
    right_arm.Movej_Cmd(RIGHT_INIT_JOINT, 20, 0, 0, True)

    # state queue
    right_pose_queue = StateQueue()

    _, cur_right_joint, cur_right_pose, _ = right_arm.Get_Current_Arm_State()
    print(f"joint:{cur_right_joint}, pose: {cur_right_pose}")
    [right_pose_queue.enqueue(p) for p in [cur_right_pose] * right_pose_queue.maxlen]

    last_right_joint = cur_right_joint

    # 用于存储运行目录
    base_log_dir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(base_log_dir, exist_ok=True)
    step = 0

    print("初始化成功！等待手柄信号...")

    while MANIPLATION_RUNNIG:
        start_time = time.time()

        _, cur_right_joint, cur_right_pose, _ = right_arm.Get_Current_Arm_State()   # pose状态
        right_griper = 0                                                            # 夹爪状态
        right_pose_queue.enqueue(cur_right_pose)
        if REALSENSE_IMAGE is None:
            print("realsence 未接入")

        if RESET_SIGNAL:
            RESET_SIGNAL = False
            right_arm.Set_Gripper_Release(500, block=False)
            right_arm.Movej_CANFD(RIGHT_INIT_JOINT, False)
            time.sleep(dT)
            continue

        if not START:
            right_arm.Movej_CANFD(cur_right_joint, False)
            time.sleep(dT)
            continue

        if max(right_pose_queue.variance()) == -100: # 使用透传时需要
            # is running, and not to get policy action
            cur_right_joint = last_right_joint
        #elif REALSENSE_IMAGE is not None or True:
        elif REALSENSE_IMAGE is not None or True:
            # getting policy action
            cur_image = REALSENSE_IMAGE
            #################################################################
            # global cur_index, frames
            # cur_image, cur_index = frames[cur_index], cur_index + 1
            #################################################################
            delta_actions = policy.process_frame(image=cur_image) #delta_actions 是 [dx, dy, dz, rx, ry, rz, gripper_value]
            num_actions = len(delta_actions)

            accum_delta_actions = [0.0]*len(cur_right_pose)
            #print(f"policy action: {delta_actions}")
            #delta_actions = delta_actions[0]
            # new pose
            for single_action in delta_actions:
                if START == False: break
                delta_right_actions, right_griper = single_action[:-1], single_action[-1]
                print(f"夹爪{right_griper}")

                for i in range(len(delta_right_actions)):
                    if delta_right_actions[i] > 0.15:
                        print("delta移动距离大于15cm, 危险, 已设置为0, 建议退出")
                        delta_right_actions[i] = 0
                    accum_delta_actions[i] += delta_right_actions[i]
            #accum_delta_actions[3] += 0.15        
            print(f"policy action: {accum_delta_actions}")
            new_right_pose = []
            for i in range(len(accum_delta_actions)):
                if abs(accum_delta_actions[i]) > 0.50:
                    accum_delta_actions[i] = 0
                    print("一帧移动大于50厘米, 危险, 已设置为0, 建议退出")
                new_value = cur_right_pose[i] + accum_delta_actions[i]
                new_right_pose.append(new_value)
            final_gripper = delta_actions[-1][-1]
            x, y, z = new_right_pose[:3]
            euler = new_right_pose[3:]
            R = euler_to_matrix(euler)
            T = [[R[0][0], R[0][1], R[0][2], x],
                [R[1][0], R[1][1], R[1][2], y],
                [R[2][0], R[2][1], R[2][2], z],
                [0,       0,       0,       1]]

            cur_right_joint = np.array(cur_right_joint) * deg2rad
            cur_right_joint = qp.sovler(cur_right_joint, T)
            cur_right_joint = [x * rad2deg for x in cur_right_joint]
            
            if START and REALSENSE_IMAGE is not None:
                # 拷贝一份当前图像
                tosave_image = REALSENSE_IMAGE.copy()

                # 每帧用同一个时间戳
                ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                # 图片保存路径
                img_path = os.path.join(base_log_dir, f"{ts}.png")
                cv2.imwrite(img_path, tosave_image)

                # 轨迹保存为 txt
                txt_path = os.path.join(base_log_dir, f"{ts}.txt")
                with open(txt_path, 'w') as f:
                    # 简单写成每个维度一行，或 JSON 形式也行
                    f.write("pose:\n")
                    for v in new_right_pose:
                        f.write(f"{v:.6f} ")
                    f.write("\n")
                    f.write(f"gripper: {final_gripper}\n")

                step += 1
            # 下发夹爪指令和位姿指令
            if final_gripper < 0.1:
                right_arm.Set_Gripper_Pick(500, 500, block=False) #0为抓
            elif final_gripper > 0.9:
                right_arm.Set_Gripper_Release(500, block=False)#1为放
            else:
                right_arm.Set_Gripper_Release(500, block=False) 
                print(f"Not in threshold")
            right_arm.Movej_Cmd(cur_right_joint, 20, 0, 0, True)
            last_right_joint = cur_right_joint
            #print(f"Executing action {idx}/{num_actions}, right joint: {cur_right_joint}, gripper: {right_griper}")
            print(f"Executing actions: right joint: {cur_right_joint}, gripper: {final_gripper}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time < dT:
            time.sleep(dT - elapsed_time)


if __name__ == '__main__':

    # start realsense get pic 创建图像采集线程
    thread1 = threading.Thread(target=get_video, args=(), daemon=True)
    thread1.start()
    wait_for_realsense()
    # policy maniplation 创建策略执行线程
    thread2 = threading.Thread(target=maniplation, args=("http://localhost:2345", "192.168.10.19", "192.168.10.18", 0.1), daemon=True)
    thread2.start()

    joystick = init_joy()
    while JOY_EVENT_RUNNING:
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                JOY_EVENT_RUNNING = False
                MANIPLATION_RUNNIG = False

        if joystick.get_button(JOY.RESET.value) == 1 and not RESET_SIGNAL:
            RESET_SIGNAL = True
            START = False
            print("reset 透传中")
        elif joystick.get_button(JOY.EXIT.value) == 1:
            MANIPLATION_RUNNIG = False
            JOY_EVENT_RUNNING = False
            print("exitting")
        
        if joystick.get_button(JOY.START_SERVER.value) == 1 and not START:
            START = True
            print(f"start policy!")
        elif joystick.get_button(JOY.STOP_SERVER.value) == 1 and START:
            START = False
            print(f"stop policy!")
        pygame.time.delay(10)
