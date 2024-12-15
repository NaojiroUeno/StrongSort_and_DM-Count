import torch
from model import vgg19
import cv2
import copy
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from sklearn.cluster import MeanShift, estimate_bandwidth

# 訓練済み重みのパスを指定
model_path = 'pretrained_model/model_nwpu.pth'

# デバイスをCPUに指定
device = torch.device('cpu')

# モデルを構築
model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))

# 推論用関数
def inference(model, image):
    # 前処理
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    input_image = input_image.to(device)

    # 推論
    with torch.set_grad_enabled(False):
        outputs, _ = model(input_image)

    # マップ取得
    result_map = outputs[0, 0].cpu().numpy()

    # マップから人数を計測
    count = torch.sum(outputs).item()

    return result_map, int(count)

# マップ可視化用関数
def create_color_map(result_map, image):
    color_map = copy.deepcopy(result_map)

    # 0～255の範囲に正規化して、疑似カラーを適用
    color_map = (color_map - color_map.min()) / (color_map.max() - color_map.min() + 1e-5)
    color_map = (color_map * 255).astype(np.uint8)
    color_map = cv2.applyColorMap(color_map, cv2.COLORMAP_JET)

    # リサイズして元画像と合成
    image_width, image_height = image.shape[1], image.shape[0]
    color_map = cv2.resize(color_map, dsize=(image_width, image_height))
    # debug_image = cv2.addWeighted(image, 0.35, color_map, 0.65, 1.0)
    debug_image = cv2.addWeighted(image, 0.30, color_map, 0.70, 1.0)

    return color_map, debug_image

def detect_crowd(image, threshold_value=0.40, y_extension=120):
    result_map, count = inference(model, image)
    color_map, img = create_color_map(result_map, image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = (img - 127.5) / 127.5
    img = 0.5 * img + 0.5
    plt.imshow(img)

    # threshold_value = 0.40
    threshold_img = img.copy()
    threshold_img[threshold_img < threshold_value] = 0
    
    # 特徴マップの作成（ここでは単純に青チャンネルを使用
    feature_map = threshold_img[:, :, 1]

    # 特徴マップの作成（ここでは単純に青チャンネルを使用）
    ##### おそらくこのままで動くがfeature_mapがそのままでいいのかは吟味する必要がある #############
    # feature_map = image[:, :, 1]

    # 黒色以外のピクセルを取得
    non_black_pixels = np.column_stack(np.where(feature_map > 0))

    # Mean Shiftクラスタリングを行うための帯域幅を推定
    bandwidth = estimate_bandwidth(non_black_pixels, quantile=0.2, n_samples=500)

    # Mean Shiftクラスタリング
    # meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # labels = meanshift.fit_predict(non_black_pixels)
    dbscan = DBSCAN(eps=100, min_samples=2)  # epsやmin_samplesは適宜調整
    labels = dbscan.fit_predict(non_black_pixels)

    # クラスタに各ピクセルを割り当て
    clustered_image = np.zeros_like(feature_map)
    for label, pixel in zip(labels, non_black_pixels):
        clustered_image[pixel[0], pixel[1]] = label + 1  # 0は背景なのでクラスタ番号を1から始める
    
    # グループを囲った画像をimage_surroundとする
    image_mask = image.copy()

    # 各クラスタの凸包を描画
    unique_labels = np.unique(labels)

    all_hull_points = []

    for label in unique_labels:
        cluster_points = non_black_pixels[labels == label]
        if len(cluster_points) > 2:
            if cluster_points.shape[0] >= 3 and np.ptp(cluster_points[:, 0]) > 0 and np.ptp(cluster_points[:, 1]) > 0:
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices]
                hull_points = hull_points[:, ::-1]  # (y, x)から(x, y)に変換

                # 凸包を描画
                hull_points = hull_points.reshape((-1, 2))  # OpenCVの形式に変換

                max_y = np.max(hull_points[:, 1])
                hull_points_extended = hull_points.copy()
                for i, point in enumerate(hull_points):
                    if point[1] == max_y:  # 最大 y 座標の点を確認
                        hull_points_extended[i, 1] += y_extension  # 下に延長

            # 凸包を描画（延長された形）
            hull_points_extended = hull_points_extended.reshape((-1, 1, 2))  # OpenCVの形式に変換
            all_hull_points.append(hull_points_extended)
    return all_hull_points