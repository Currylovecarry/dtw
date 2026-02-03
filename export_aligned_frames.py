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
