# -*- coding: utf-8 -*-
import threading
import numpy as np
from enum import Enum
from collections import deque
# realsense
import pyrealsense2 as rs
# joy
import pygame
import sys

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


def get_vedio():
    global REALSENSE_IMAGE

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 设置视频流的参数

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue
            REALSENSE_IMAGE = np.asanyarray(color_frame.get_data())
    finally:
        pipeline.stop()

def maniplation(policy_url="http://localhost:2345", right_ram_url="192.168.10.19", left_arm_url="192.168.10.18", dT=0.1):

    global RESET_SIGNAL, REALSENSE_IMAGE, MANIPLATION_RUNNIG, START

    policy = PolicyClient(base_url=policy_url)
    #print(f"{policy_url} - policy connection state: {policy.reset()}")

    # 实例化逆解库             
    qp = QPIK("RM75B", dT)
    # qp.set_install_angle([90, 180, 0], 'deg')
    qp.set_install_angle([0, -90, 0], 'deg')

    qp.set_work_cs_params([0, 0, 0, 0, -1.570, 0])
    qp.set_tool_cs_params([0, 0, 0, 0, 0, 0])

    #qp.set_joint_limit_max([ 178,  130,  178,  135,  178,  128, 360], 'deg')
    #qp.set_joint_limit_min([-178, -130, -178, -135, -178, -128, -360], 'deg')
    qp.set_joint_limit_max([ -40,  -10,  178,  135,  178,  128, 360], 'deg')
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

    # state queue
    left_pose_queue = StateQueue()

    _, cur_left_joint, cur_left_pose, _ = left_arm.Get_Current_Arm_State()

    [left_pose_queue.enqueue(p) for p in [cur_left_pose] * left_pose_queue.maxlen]

    last_left_joint = cur_left_joint

    print("初始化成功！等待手柄信号...")

    while MANIPLATION_RUNNIG:
        start_time = time.time()

        _, cur_left_joint, cur_left_pose, _ = left_arm.Get_Current_Arm_State()   # pose状态
        left_griper = 0                                                            # 夹爪状态
        left_pose_queue.enqueue(cur_left_pose)
        if REALSENSE_IMAGE is None:
            print("realsence 未接入")

        if RESET_SIGNAL:
            RESET_SIGNAL = False
            left_arm.Movej_CANFD(LEFT_INIT_JOINT, False)
            time.sleep(dT)
            continue

        if not START:
            left_arm.Movej_CANFD(cur_left_joint, False)
            time.sleep(dT)
            continue

        if max(left_pose_queue.variance()) == -100: # 使用透传时需要
            # is running, and not to get policy action
            cur_left_joint = last_left_joint
        elif REALSENSE_IMAGE is not None or True:
            # getting policy action
            cur_image = REALSENSE_IMAGE
            #################################################################
            # global cur_index, frames
            # cur_image, cur_index = frames[cur_index], cur_index + 1
            #################################################################
            delta_actions = policy.process_frame(image=cur_image) #delta_actions 是 [dx, dy, dz, rx, ry, rz, gripper_value]
            print(f"policy action: {delta_actions}")
            #####test##########
            #delta_left_actions = delta_actions[0]
            #print(type(delta_left_actions), type(delta_left_actions[0]))
#           ##########
            # new pose
            
            ##Add
            delta_actions = delta_actions[0]
            ##

            delta_left_actions, left_griper = delta_actions[:-1], delta_actions[-1]
            print(len(cur_left_pose), type(cur_left_pose[0]))
            print(len(delta_left_actions), type(delta_left_actions[0]))
            new_left_pose = [cur_left_pose[i] + delta_left_actions[i] for i in range(len(delta_left_actions))]
            x, y, z = new_left_pose[:3]
            euler = new_left_pose[3:]
            R = euler_to_matrix(euler)
            T = [[R[0][0], R[0][1], R[0][2], x],
                [R[1][0], R[1][1], R[1][2], y],
                [R[2][0], R[2][1], R[2][2], z],
                [0,       0,       0,       1]]

            cur_left_joint = np.array(cur_left_joint) * deg2rad
            cur_left_joint = qp.sovler(cur_left_joint, T)
            cur_left_joint = [x * rad2deg for x in cur_left_joint]
        # 设置夹爪
        # 推荐阈值处理逻辑
        if left_griper < 0.1:
            left_arm.Set_Gripper_Pick(500, 500, block=False)
        elif left_griper > 0.9:
            left_arm.Set_Gripper_Release(500, block=False)
        else:
            left_arm.Set_Gripper_Release(500, block=False) 
            print(f"Not in threshold: {left_griper}")

        # 设置位姿
        left_arm.Movej_Cmd(cur_left_joint, 20, 0, 0, True)
        last_left_joint = cur_left_joint

        end_time = time.time()
        elapsed_time = end_time - start_time
        if elapsed_time < dT:
            time.sleep(dT - elapsed_time)


if __name__ == '__main__':

    # start realsense get pic 创建图像采集线程
    thread1 = threading.Thread(target=get_vedio, args=(), daemon=True)
    thread1.start()

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
