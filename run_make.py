# -*- coding: utf-8 -*-
# 用于双臂插吸管任务的特化脚本
# 7.31修改，更新了reset逻辑，加入了锁定手臂的功能
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
FLAG = 0
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
    left_arm.Movej_Cmd(LEFT_INIT_JOINT, 20, 0, 0, True)
    right_arm = Arm(RM75, right_arm_url)
    right_arm.Set_Gripper_Release(500, block=False)
    right_arm.Movej_Cmd(RIGHT_INIT_JOINT, 20, 0, 0, True)

    # state queue
    left_pose_queue = StateQueue()
    right_pose_queue = StateQueue()

    _, cur_left_joint, cur_left_pose, _ = left_arm.Get_Current_Arm_State()
    _, cur_right_joint, cur_right_pose, _ = right_arm.Get_Current_Arm_State()

    print(f"joint:{cur_right_joint}, pose: {cur_right_pose}")
    
    [left_pose_queue.enqueue(cur_left_pose) for _ in range(left_pose_queue.maxlen)]
    [right_pose_queue.enqueue(cur_right_pose) for _ in range(right_pose_queue.maxlen)]  

    last_right_joint = cur_right_joint
    last_left_joint = cur_left_joint

    arm_locked = False
    arm_locker = None

    # 用于存储运行目录
    base_log_dir = os.path.join('logs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(base_log_dir, exist_ok=True)
    step = 0
    print("初始化成功！等待手柄信号...")


    while MANIPLATION_RUNNIG:
        start_time = time.time()

         # 1. 读当前左右末端 pose
        _, _, cur_left_pose,  _ = left_arm.Get_Current_Arm_State()
        _, _, cur_right_pose, _ = right_arm.Get_Current_Arm_State()
        right_gripper = 0                                                          
        
        if RESET_SIGNAL:
            RESET_SIGNAL = False
            left_arm.Set_Gripper_Release(500, block=False)
            left_arm.Movej_CANFD(LEFT_INIT_JOINT, False)
            right_arm.Set_Gripper_Release(500, block=False)
            right_arm.Movej_CANFD(RIGHT_INIT_JOINT, False)
            time.sleep(dT)
            # reset同步更新内部状态
            cur_left_joint  = LEFT_INIT_JOINT.copy()
            last_left_joint = LEFT_INIT_JOINT.copy()
            cur_right_joint  = RIGHT_INIT_JOINT.copy()
            last_right_joint = RIGHT_INIT_JOINT.copy()
            arm_locked = False
            continue

        if not START:
            left_arm.Movej_CANFD(cur_left_joint, False)
            right_arm.Movej_CANFD(cur_right_joint, False)
            time.sleep(dT)
            arm_locked = False
            continue
        
        if REALSENSE_IMAGE is None:
            print("realsense 未接入")
            continue
        
        if max(right_pose_queue.variance()) == -100: # 使用透传时需要
            # is running, and not to get policy action
            cur_left_joint = last_left_joint
            cur_right_joint = last_right_joint
        
        elif REALSENSE_IMAGE is not None or True:
            delta_actions = np.array(policy.process_frame(image=REALSENSE_IMAGE), dtype=np.float32) #delta_actions 是 [dx, dy, dz, rx, ry, rz, gripper_value]
            accum_l = np.zeros(6)
            accum_r = np.zeros(6)
            left_gripper = 0
            right_gripper = 0
            
            for single_action in delta_actions:
                if START == False: break
                delta_left_actions, delta_right_actions, action_arm = single_action[:7], single_action[7:14], single_action[14]# 前7列：左臂，后7列：右臂
                left_gripper, right_gripper = delta_left_actions[-1], delta_right_actions[-1]   #夹爪直接取最后的值
                accum_l += delta_left_actions[:-1]
                accum_r += delta_right_actions[:-1]
            for i in range(len(accum_l)):
                if abs(accum_l[i]) > 0.50:
                    accum_l[i] = 0
                    print("左臂一帧移动大于50厘米, 危险, 已设置为0, 建议退出")
                if abs(accum_r[i]) > 0.50:
                    accum_r[i] = 0
                    print("右臂一帧移动大于50厘米, 危险, 已设置为0, 建议退出")
            # 计算新的末端 pose（xyz + rpy）
            _, _, cur_left_pose, _  = left_arm.Get_Current_Arm_State()
            _, _, cur_right_pose, _ = right_arm.Get_Current_Arm_State()
            if action_arm < 0.5:
                accum_r[:] = 0
                right_gripper = 1
            else:
                accum_l[:] = 0
                left_gripper = 1
            new_left  = cur_left_pose  + accum_l
            new_right = cur_right_pose + accum_r
            
            # 构造变换矩阵
            x, y, z = new_left[:3]
            euler = new_left[3:]
            R = euler_to_matrix(euler)
            T_left = [[R[0][0], R[0][1], R[0][2], x],
                [R[1][0], R[1][1], R[1][2], y],
                [R[2][0], R[2][1], R[2][2], z],
                [0,       0,       0,       1]]
            x, y, z = new_right[:3]
            euler = new_right[3:]
            R = euler_to_matrix(euler)
            T_right = [[R[0][0], R[0][1], R[0][2], x], 
                [R[1][0], R[1][1], R[1][2], y],
                [R[2][0], R[2][1], R[2][2], z],
                [0,       0,       0,       1]]
            
            left_j  = np.deg2rad(left_arm.Get_Current_Arm_State()[1])
            left_j  = qp.sovler(left_j, T_left)
            left_j  = np.rad2deg(left_j).tolist()
            right_j = np.deg2rad(right_arm.Get_Current_Arm_State()[1])
            right_j = qp.sovler(right_j, T_right)
            right_j = np.rad2deg(right_j).tolist()

            # 下发夹爪指令和位姿指令
            if left_gripper < 0.5:
                left_arm.Set_Gripper_Pick(500, 500, block=False) #0为抓
            else:
                left_arm.Set_Gripper_Release(500, block=False)#1为放
            if right_gripper < 0.5:
                right_arm.Set_Gripper_Pick(500, 500, block=False) #0为抓
            else:
                right_arm.Set_Gripper_Release(500, block=False)#1为放
            
            if action_arm < 0.5:
                left_arm.Movej_Cmd(left_j,  20, 0, 0, True)
                cur_left_joint = left_j
            else:
                right_arm.Movej_Cmd(right_j, 20, 0, 0, True)
                cur_right_joint = right_j
            
            last_left_joint = cur_left_joint
            last_right_joint = cur_right_joint
            #print(f"Executing action {idx}/{num_actions}, right joint: {cur_right_joint}, gripper: {right_gripper}")
            print(f"Executing actions: right joint: {cur_right_joint}, gripper: {right_gripper}")
            print(f"left joint: {cur_left_joint}, gripper: {left_gripper}")

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
