"""
utils.py
---------
Utility functions for drawing bounding boxes, counting objects, and calculating FPS.
"""

import cv2
import time
import numpy as np
import random
from typing import List, Dict, Tuple


class Visualizer:
    def __init__(self, class_ids, class_names):
        self.class_ids = class_ids
        self.class_names = class_names
        self.counts = {cls: 0 for cls in class_ids}
        self.tracked_ids = set()

    def draw_tracks(self, frame, tracks, conf_thresh=0.4):
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = map(int, track['bbox'])
            cls_id = track['class_id']
            license_text = track['license']

            # 绘制 bounding box
            color = self._get_color(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制轨迹ID和类别
            label = f"ID: {track_id} {self.class_names[cls_id]}"
            if license_text:
                label += f" | License: {license_text}"
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # 更新计数
            if track_id not in self.tracked_ids:
                self.tracked_ids.add(track_id)
                self.counts[cls_id] += 1

    def draw_counts(self, frame):
        # 绘制各类别计数
        y_offset = 30
        for cls_id, count in self.counts.items():
            text = f"{self.class_names[cls_id]}: {count}"
            cv2.putText(
                frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            y_offset += 30

    @staticmethod
    def _get_color(track_id):
        # 为不同轨迹生成唯一颜色
        seed = int(track_id)
        np.random.seed(seed)
        return tuple(map(int, np.random.randint(0, 255, 3)))
