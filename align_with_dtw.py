from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def align_pose_sequences(seq1, seq2):
    # fastdtw 返回最短路径，格式是 [(idx1, idx2), ...]
    distance, path = fastdtw(seq1, seq2, dist=euclidean)
    return path