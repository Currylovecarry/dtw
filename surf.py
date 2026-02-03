from extract_pose_sequence import extract_pose_sequence
from align_with_dtw import align_pose_sequences
from save_alignment_to_json import save_alignment_path
import matplotlib.pyplot as plt
from export_aligned_video import export_aligned_video
import cv2
import mediapipe as mp
import numpy as np


# 加载两个视频并提取骨架数据
video1_path = "real_exam2.mp4"
video2_path = "exam.mp4"

print("提取 video1...")
seq1 = extract_pose_sequence(video1_path)
print(f"video1 提取完成，帧数: {len(seq1)}")

print("提取 video2...")
seq2 = extract_pose_sequence(video2_path)
print(f"video2 提取完成，帧数: {len(seq2)}")

# 对齐
print("开始DTW对齐...")
alignment_path = align_pose_sequences(seq1, seq2)
print(f"对齐完成，共 {len(alignment_path)} 对帧。")

# 打印前10帧对应关系
print("\n前10帧对齐结果:")
for i, (a, b) in enumerate(alignment_path[:10]):
    print(f"Video1 frame {a} ↔ Video2 frame {b}")

# 保存JSON
save_alignment_path(alignment_path, "alignment.json")

# 可视化匹配路径
x, y = zip(*alignment_path)
plt.figure(figsize=(8,6))
plt.plot(x, y)
plt.xlabel('Video 1 Frame Index')
plt.ylabel('Video 2 Frame Index')
plt.title('DTW Alignment Path')
plt.grid()
plt.tight_layout()
plt.show()

print("\n处理完成，对齐结果已保存到 alignment.json")

# 导出对齐后视频
export_aligned_video(video1_path, video2_path, alignment_path, "aligned_output.mp4")

# 可视化匹配路径
x, y = zip(*alignment_path)
plt.figure(figsize=(8,6))
plt.plot(x, y)
plt.xlabel('Video 1 Frame Index')
plt.ylabel('Video 2 Frame Index')
plt.title('DTW Alignment Path')
plt.grid()
plt.tight_layout()
plt.show()

# 保存对齐后的视频
export_aligned_video(video1_path, video2_path, alignment_path, "aligned_output.mp4")

print("\n全部处理完成！对齐结果已保存。")


from export_aligned_frames import export_aligned_frames

export_aligned_frames(video1_path, video2_path, alignment_path, output_dir="output_frames")

from extract_keypoints_to_json import extract_keypoints_to_json

# 提取每帧的33个关键点并保存为JSON文件
extract_keypoints_to_json("real_exam2.mp4", output_dir="keypoints_json")
extract_keypoints_to_json("exam.mp4", output_dir="keypoints2_json")


def extract_pose_sequence(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

    cap = cv2.VideoCapture(video_path)
    frame_features = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Flatten all landmark coordinates into a 1D vector
            landmarks = results.pose_landmarks.landmark
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
        else:
            keypoints = np.zeros(33 * 3)  # empty pose

        frame_features.append(keypoints)

    cap.release()
    return np.array(frame_features)


from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def align_pose_sequences(seq1, seq2):
    # fastdtw 返回最短路径，格式是 [(idx1, idx2), ...]
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    return path



import json

def save_alignment_path(alignment_path, output_path="alignment.json"):
    """
    alignment_path: List of (i, j) tuples
    output_path:    JSON file to save
    """
    # 转换为字典列表
    alignment_data = [
        {"video1_frame": int(i), "video2_frame": int(j)}
        for i, j in alignment_path
    ]

    with open(output_path, "w") as f:
        json.dump(alignment_data, f, indent=2)

    print(f"Alignment saved to {output_path}")


import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

def draw_pose(image, pose_landmarks):
    """在frame上绘制关键点"""
    if not pose_landmarks:
        return image

    annotated_image = image.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(
        annotated_image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
    )
    return annotated_image

def export_aligned_video(video1_path, video2_path, alignment_path, output_path="aligned_output.mp4"):
    # Mediapipe初始化
    pose = mp_pose.Pose(static_image_mode=True)

    # 打开视频
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 获取帧宽高
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 输出视频宽高（左右拼接）
    out_width = width1 + width2
    out_height = max(height1, height2)

    # 定义视频写入
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 25  # 根据需要改帧率
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"开始导出对齐视频，共 {len(alignment_path)} 帧...")

    for idx, (frame_idx1, frame_idx2) in enumerate(alignment_path):
        # 定位到帧
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx2)

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print(f"警告: 第{idx}帧读取失败，跳过。")
            continue

        # Mediapipe处理
        rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        results1 = pose.process(rgb1)
        results2 = pose.process(rgb2)

        # 绘制骨架
        frame1 = draw_pose(frame1, results1.pose_landmarks)
        frame2 = draw_pose(frame2, results2.pose_landmarks)

        # 如果帧高不同，resize
        if height1 != height2:
            frame2 = cv2.resize(frame2, (width2, height1))

        # 拼接
        combined_frame = np.hstack((frame1, frame2))

        out.write(combined_frame)

        if idx % 50 == 0:
            print(f"已处理 {idx}/{len(alignment_path)} 帧")

    # 释放资源
    cap1.release()
    cap2.release()
    out.release()
    pose.close()

    print(f"\n导出完成！合成视频保存在 {output_path}")


import cv2
import numpy as np
import mediapipe as mp
import os

mp_pose = mp.solutions.pose

def draw_pose(image, pose_landmarks):
    """在frame上绘制关键点"""
    if not pose_landmarks:
        return image

    annotated_image = image.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(
        annotated_image,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
    )
    return annotated_image

def export_aligned_frames(video1_path, video2_path, alignment_path, output_dir="output_frames"):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # Mediapipe初始化
    pose = mp_pose.Pose(static_image_mode=True)

    # 打开视频
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    # 获取帧高
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"开始导出对齐帧，共 {len(alignment_path)} 帧...")

    for idx, (frame_idx1, frame_idx2) in enumerate(alignment_path):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_idx1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx2)

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print(f"警告: 第{idx}帧读取失败，跳过。")
            continue

        # Mediapipe处理
        rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        results1 = pose.process(rgb1)
        results2 = pose.process(rgb2)

        # 绘制骨架
        frame1 = draw_pose(frame1, results1.pose_landmarks)
        frame2 = draw_pose(frame2, results2.pose_landmarks)

        # 如果帧高不同，resize
        if height1 != height2:
            frame2 = cv2.resize(frame2, (frame2.shape[1], height1))

        # 拼接
        combined_frame = np.hstack((frame1, frame2))

        # 保存
        filename = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
        cv2.imwrite(filename, combined_frame)

        if idx % 50 == 0:
            print(f"已保存 {idx}/{len(alignment_path)} 帧")

    # 释放资源
    cap1.release()
    cap2.release()
    pose.close()

    print(f"\n导出完成！所有帧已保存到 {output_dir}")

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

