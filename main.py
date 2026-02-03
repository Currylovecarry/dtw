from extract_pose_sequence import extract_pose_sequence
from align_with_dtw import align_pose_sequences
from save_alignment_to_json import save_alignment_path
import matplotlib.pyplot as plt
from export_aligned_video import export_aligned_video

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