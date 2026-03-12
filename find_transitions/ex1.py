import cv2
import numpy as np
import matplotlib.pyplot as plt

def load(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        # recupere les frames
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()
    return frames

def compute_histogram(frame):
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256]).reshape(-1)
    hist = hist.astype(float)

    return hist

def compute_cdf(hist):
    cdf = np.cumsum(hist)
    return cdf

def compute_distance(hist1, hist2):
        return np.sum(np.abs(hist1 - hist2))


def detect_scene_1(frames):
    distances = []
    for i in range (len(frames)-1):
        hist1 = compute_histogram(frames[i])
        hist2 = compute_histogram(frames[i+1])
        distance = compute_distance(hist1, hist2)
        distances.append(distance)

    max_diff_idx = np.argmax(distances)

    return max_diff_idx, max_diff_idx + 1

def detect_scene_2(frames):
    distances = []
    for i in range (len(frames)-1):
        hist1 = compute_histogram(frames[i])
        hist2 = compute_histogram(frames[i+1])
        cdf1 = compute_cdf(hist1)
        cdf2 = compute_cdf(hist2)
        distance = compute_distance(cdf1, cdf2)
        distances.append(distance)

    max_diff_idx = np.argmax(distances)

    return max_diff_idx, max_diff_idx + 1

def visualize_scene(frames,j):
    distances = []
    for i in range(len(frames) - 1):
        hist1 = compute_histogram(frames[i])
        hist2 = compute_histogram(frames[i + 1])
        # Tu peux changer pour utiliser le CDF ici si c'est ta métrique principale
        # cdf1, cdf2 = compute_cdf(hist1), compute_cdf(hist2)
        # distance = compute_distance(cdf1, cdf2)
        distance = compute_distance(hist1, hist2)
        distances.append(distance)

    plt.figure(figsize=(10, 4))
    plt.plot(distances, marker='o', markersize=3)
    plt.title(f"variation inter-frame (histo) video:{j}")
    plt.xlabel("Index of frame")
    plt.ylabel("Distance")
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.show()

def visualize_scene2(frames,j):
    distances = []
    for i in range(len(frames) - 1):
        hist1 = compute_histogram(frames[i])
        hist2 = compute_histogram(frames[i + 1])
        # Tu peux changer pour utiliser le CDF ici si c'est ta métrique principale
        cdf1, cdf2 = compute_cdf(hist1), compute_cdf(hist2)
        distance = compute_distance(cdf1, cdf2)
        distances.append(distance)

    plt.figure(figsize=(10, 4))
    plt.plot(distances, marker='o', markersize=3)
    plt.title(f"variation inter-frame (cumulative histo) video:{j}")
    plt.xlabel("Index of frame")
    plt.ylabel("Distance")
    plt.grid(True, linestyle=":", linewidth=0.8)
    plt.tight_layout()
    plt.show()



def main(video_path, video_type):
    """
    Main entry point for ex1
    :param video_path: path to video file
    :param video_type: category of the video (either 1 or 2)
    :return: a tuple of integers representing the frame number for which the scene cut was detected (i.e. the last frame index of the first scene and the first frame index of the second scene)
    """
    frames = load(video_path)
    if video_type == 1:
        cut_frame1 , cut_frame2 = detect_scene_1(frames)
    elif video_type == 2:
        cut_frame1, cut_frame2 = detect_scene_2(frames)
    visualize_scene(frames,4)
    visualize_scene2(frames,4)
    return int(cut_frame1), int(cut_frame2)



if __name__ == '__main__':
    video_path = 'video4_category2.mp4'
    video_type = 1
    result = main(video_path, video_type)
    print("the result is ",result)