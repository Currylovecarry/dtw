import cv2
import numpy as np
import mediapipe as mp
import json
import os

mp_pose = mp.solutions.pose

def extract_keypoints_to_json(video_path, output_dir="keypoints_json"):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # Mediapipe初始化
    pose = mp_pose.Pose(static_image_mode=True)

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"开始提取关键点，共 {total_frames} 帧...")

    for idx in range(total_frames):
        # 定位到该帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            print(f"警告: 第{idx}帧读取失败，跳过。")
            continue

        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 获取姿势检测结果
        results = pose.process(rgb_frame)

        # 如果该帧包含关键点
        if results.pose_landmarks:
            keypoints = {}
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints[f"landmark_{i}"] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                }

            # 保存为JSON文件
            filename = os.path.join(output_dir, f"frame_{idx:04d}.json")
            with open(filename, "w") as json_file:
                json.dump(keypoints, json_file, indent=4)

            print(f"已保存第 {idx} 帧关键点数据：{filename}")

    # 释放资源
    cap.release()
    pose.close()

    print(f"\n关键点提取完成，数据已保存到 {output_dir} 文件夹")