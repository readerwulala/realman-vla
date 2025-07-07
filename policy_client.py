# 这段代码是用来与一个远程服务器进行交互的客户端，主要用于处理图像和动作指令。
# 它可以发送图像和文本指令到服务器，并接收处理后的动作
import json
import requests
import cv2

class PolicyClient:
    def __init__(self, base_url='http://localhost:2345', max_frames=6, temperature=0.6, language_instruction="Pick up the straw and insert it into the cup on the table"):
        self.base_url = base_url
        self.max_frames = max_frames
        self.temperature = temperature

        self.language_instruction = language_instruction

        self.decode_answer = False

        print(f"Current temperature value is: {temperature}.")

        # for test 
        self.test_demo = self.load_test_demo()
        self.test_index = 0

        # last action
        self.last_action = [0, 0, 0, 0, 0, 0, -1]


    def reset(self):
        self.test_index = 0
        return requests.post(self.base_url+"/reset").json().get('response')

    def process_frame(self, **kwargs):
        kwargs = self.preprocess_image(**kwargs)
        # 显示图像
        cv2.imshow('transferred image frame', kwargs['image'])
        cv2.waitKey(1)

        #上传图片和promot至server
        encoded_imgs = {}
        for name, img in kwargs.items():
            if name not in ["image"]: continue
            if img is not None:
                ret, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                assert ret, "Image encode failed"
                encoded_img = encoded_img.tobytes()
            else:
                encoded_img = None
            encoded_imgs.update({name: encoded_img})
        ret = requests.post(
                self.base_url+"/process_frame",
                data={"text": self.language_instruction, "temperature": self.temperature},
                files=encoded_imgs
            )
        action = ret.json().get('response')
        
        #读取本地数据并replay动作测试，使用时注释掉上方上传服务器代码
        # action = self.test_demo[self.test_index] + [0]
        # self.test_index +=1
        return self.action_decoder(action) if self.decode_answer else action

    def decode_action(self, str_action=""):
        # return [0, 0, 0, 0, 0, 0, 1]
        action = []
        action += self.test_demo[self.test_index] + [1]
        if self.test_index < len(self.test_demo) - 1:
            self.test_index += 1
        return action

    def load_test_demo(self):
        # test demo
        with open('demo.json', 'r', encoding='utf-8') as file:
            joint = json.load(file)
        joint = joint[:400]
        joint = [j['right_arm_pose'] for j in joint]  # right_arm_joint  right_arm_pose
        self.init_pose = joint[0]
        cur, delta_joint = joint[0], []
        for j in joint:
            delta_joint.append([j[i] - cur[i] for i in range(len(j))])
            cur = j
        return delta_joint  # joint  delta_joint

    def action_decoder(self, act):
        print(act)
        act = act.split(" ")

        if len(act) != 7:
            return [0, 0, 0, 0, 0, self.last_action[-1]]
        # act_range = {
        #     "world_vector_x": [-0.04, 0.03],
        #     "world_vector_y": [-0.01, 0.02],
        #     "world_vector_z": [-0.02, 0.02],
        #     "rotation_delta_x": [-0.63, 0.86],
        #     "rotation_delta_y": [-0.32, 0.26],
        #     "rotation_delta_z": [-1.31, 1.5],
        #     "gripper_closedness_action": [0, 1]
        # }
        act_range = {
            "world_vector_x": [-0.24, 0.24],
            "world_vector_y": [-0.5, 0.5],
            "world_vector_z": [-0.3, 0.3],
            "rotation_delta_x": [-1.89, 1.86],
            "rotation_delta_y": [-1.25, 1.25],
            "rotation_delta_z": [-0.52, 0.53],
            "gripper_closedness_action": [0, 1]
        }
        wx_min = act_range['world_vector_x'][0]
        wy_min = act_range['world_vector_y'][0]
        wz_min = act_range['world_vector_z'][0]
        wx_range = act_range['world_vector_x'][1] - act_range['world_vector_x'][0]
        wy_range = act_range['world_vector_y'][1] - act_range['world_vector_y'][0]
        wz_range = act_range['world_vector_z'][1] - act_range['world_vector_z'][0]
            
        rdx_min = act_range['rotation_delta_x'][0]  # rd == rotation_delta
        rdy_min = act_range['rotation_delta_y'][0]
        rdz_min = act_range['rotation_delta_z'][0]
        rdx_range = act_range['rotation_delta_x'][1] - act_range['rotation_delta_x'][0]
        rdy_range = act_range['rotation_delta_y'][1] - act_range['rotation_delta_y'][0]
        rdz_range = act_range['rotation_delta_z'][1] - act_range['rotation_delta_z'][0]

        gc_min = act_range['gripper_closedness_action'][0]  # gc == gripper_closedness
        gc_range = act_range['gripper_closedness_action'][1] - act_range['gripper_closedness_action'][0]

        act[:1] = [wx_min + int(a) * wx_range / 254 for a in act[:1]]
        act[1:2] = [wy_min + int(a) * wy_range / 254 for a in act[1:2]]
        act[2:3] = [wz_min + int(a) * wz_range / 254 for a in act[2:3]]
        act[3:4] = [rdx_min + int(a) * rdx_range / 254 for a in act[3:4]]
        act[4:5] = [rdy_min + int(a) * rdy_range / 254 for a in act[4:5]]
        act[5:6] = [rdz_min + int(a) * rdz_range / 254 for a in act[5:6]]
        act[6:] = [(gc_min + int(a) * gc_range / 254) for a in act[6:]]

        self.last_action = act
        return act

    #裁剪图片
    def preprocess_image(self, **kwargs):
        img = kwargs['image']
        h, w, _ = img.shape
        size = min(h, w)
        if h > w:
            top = (h - size) // 2
            bottom = top + size
            left, right = 0, w
        else:
            left = (w - size) // 2
            right = left + size
            top, bottom = 0, h

        kwargs['image'] = img[top:bottom, left:right]
        return kwargs
