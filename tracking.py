"""
tracking.py
------------
Handles multi-object tracking using DeepSort
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track  # 导入原生Track类
from deep_sort_realtime.deep_sort import nn_matching  # 导入余弦距离计算
import numpy as np
from typing import List, Tuple, Dict
from fuzzywuzzy import fuzz  # 需安装: pip install fuzzywuzzy python-Levenshtein

class Tracker:
    def __init__(self, max_age: int=90, max_cosine_distance: float=0.3, n_init: int=3):
        """
        Initiates tracking using DeepSort tracker.

        Args:
            max_age(int): Maximum number of frames to keep lost tracks alive.
            max_cosine_distance(float): Threshold of object's similarity.
            n_init(int): Number of consecutive detections before track is confirmed.
        """
        self.tracker = DeepSort(max_age=max_age, max_cosine_distance=max_cosine_distance, n_init=n_init)

    def update_tracker(self, detections: List, frame: np.ndarray) -> List[Tuple[List[int], float, int]]:
        """
        Updates tracker with new detections.

        Args:
            detections(List): List of detected bounding boxes with format ([x,y,w,h], conf, cls)
            frame(np.ndarray): Current video frames.

        Returns:
            List: Active tracks after update. Each track has:
                    - track_id(int)
                    - bbox(list[int]): [x1, y1, x2, y2]
                    - class_id(int)
                    - confidence(float)
        """
        return self.tracker.update_tracks(detections, frame=frame)


# 定义余弦距离度量（与原生DeepSORT保持一致）
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", 0.3, 100  # 对应DeepSORT的max_cosine_distance和nn_budget
)


class PlateTracker:
    def __init__(self, max_age: int = 90, n_init: int = 3):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            # 不使用override_track_class，后续通过其他方式关联车牌信息
        )
        # 存储跟踪ID对应的车牌信息: {track_id: {'plate_no': '...', 'color': '...', 'conf': ...}}
        self.track_plate_info: Dict[int, Dict] = {}
        # 临时存储当前帧的车牌识别结果
        self.current_plates: List[Dict] = []
        # 存储跟踪ID与自定义PlateTrack的映射
        self.plate_tracks: Dict[int, "PlateTrack"] = {}

    def update_tracker(self, detections: List, frame: np.ndarray, plate_results: List[Dict]) -> List:
        """
        更新跟踪器，融合车牌信息
        :param plate_results: 包含车辆框和车牌信息的列表
        """
        self.current_plates = plate_results  # 保存当前帧车牌结果
        # 调用原生DeepSORT更新，获取原生Track列表
        native_tracks = self.tracker.update_tracks(detections, frame=frame)
        # 包装原生Track为PlateTrack，并关联车牌信息
        self._wrap_native_tracks(native_tracks)
        # 转换为统一输出格式
        return self._format_track_results(native_tracks)

    def _wrap_native_tracks(self, native_tracks):
        """将原生Track包装为自定义PlateTrack，并关联车牌"""
        for native_track in native_tracks:
            if not native_track.is_confirmed():
                continue
            track_id = native_track.track_id
            # 若不存在则创建PlateTrack实例，存在则更新原生Track引用
            if track_id not in self.plate_tracks:
                self.plate_tracks[track_id] = PlateTrack(native_track)
            else:
                self.plate_tracks[track_id].track = native_track

            # 关联车牌信息到PlateTrack
            plate_info = self.track_plate_info.get(track_id)
            self.plate_tracks[track_id].plate_info = plate_info

        # 清理失效的跟踪（已删除的track_id）
        valid_track_ids = {t.track_id for t in native_tracks if t.is_confirmed()}
        self.plate_tracks = {tid: pt for tid, pt in self.plate_tracks.items() if tid in valid_track_ids}

    def _associate_plate_to_tracks(self, native_tracks):
        """将车牌识别结果关联到跟踪ID（原生Track）"""
        for track in native_tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            track_bbox = track.to_ltrb()  # (x1, y1, x2, y2)

            # 寻找与跟踪框IOU最高的车牌
            best_iou = 0.3  # 最小IOU阈值
            best_plate = None
            for pr in self.current_plates:
                veh_bbox = pr["vehicle_bbox"]
                iou = self._bbox_iou(track_bbox, veh_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_plate = pr["plate_info"]

            if best_plate:
                # 更新跟踪ID对应的车牌信息
                self.track_plate_info[track_id] = {
                    "plate_no": best_plate["plate_no"],
                    "color": best_plate["plate_color"],
                    "conf": best_plate["detect_conf"] * best_plate["rec_conf"]
                }

    def _format_track_results(self, native_tracks) -> List:
        """格式化跟踪结果为统一输出格式"""
        result_tracks = []
        for track in native_tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()  # (x1, y1, x2, y2)
            result_tracks.append([
                ltrb[0], ltrb[1], ltrb[2], ltrb[3],
                track_id, track.get_det_class(), track.det_conf
            ])
        return result_tracks

    @staticmethod
    def _bbox_iou(box1: Tuple, box2: Tuple) -> float:
        """计算两个边界框的IOU"""
        x1, y1, x2, y2 = box1
        a1, b1, a2, b2 = box2
        inter_x1 = max(x1, a1)
        inter_y1 = max(y1, b1)
        inter_x2 = min(x2, a2)
        inter_y2 = min(y2, b2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (a2 - a1) * (b2 - b1)
        return inter_area / (area1 + area2 - inter_area + 1e-5)


class PlateTrack:
    """自定义跟踪类，包装原生Track以支持车牌特征"""

    def __init__(self, native_track: Track):
        self.track = native_track  # 原生Track实例
        self.plate_info = None  # 车牌信息

    def __getattr__(self, name):
        """代理原生Track的所有方法和属性"""
        return getattr(self.track, name)

    def calculate_match_distance(self, other_plate_track: "PlateTrack") -> float:
        """
        计算两个PlateTrack的匹配距离（融合外观、IOU、车牌）
        替代之前的appearance_distance，基于原生DeepSORT的余弦距离实现
        """
        # 1. 外观余弦距离（原生DeepSORT的核心度量，0-1，越小越相似）
        appearance_dist = 1.0
        if self.track.features and other_plate_track.track.features:
            # 取最新的特征向量计算余弦距离
            feat1 = self.track.features[-1].reshape(1, -1)
            feat2 = other_plate_track.track.features[-1].reshape(1, -1)
            appearance_dist = metric.distance(feat1, feat2)[0][0]

        # 2. IOU距离（0-1，越小越相似）
        iou = PlateTracker._bbox_iou(self.to_ltrb(), other_plate_track.to_ltrb())
        iou_dist = 1 - iou

        # 3. 车牌距离（0-1，越小越相似）
        plate_dist = 1.0
        if self.plate_info and other_plate_track.plate_info:
            plate_similarity = fuzz.ratio(
                self.plate_info["plate_no"],
                other_plate_track.plate_info["plate_no"]
            ) / 100.0
            plate_dist = 1 - plate_similarity

        # 加权融合（可根据场景调整权重）
        return 0.5 * appearance_dist + 0.3 * iou_dist + 0.2 * plate_dist