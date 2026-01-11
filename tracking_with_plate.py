"""
tracking_with_plate.py
----------------------
多目标车辆跟踪（分离车牌检测识别，仅负责跟踪逻辑）
适配 deep_sort_realtime 库原生接口，支持车牌优先匹配
"""

from deep_sort_realtime.deepsort_tracker import DeepSort as BaseDeepSort
from deep_sort_realtime.deep_sort.track import Track
from deep_sort_realtime.deep_sort.detection import Detection
from Levenshtein import ratio  # 需安装 python-Levenshtein
import numpy as np
from typing import List, Dict, Iterable
import logging
import time  # 确保导入time模块（若已导入可忽略）


# 自定义常量：车牌信息在 Detection.others 中的存储键
LICENSE_KEY = "license_text"

class Tracker:
    def __init__(self, max_age=90, max_cosine_distance=0.3, n_init=3, license_match_thresh=0.8):
        """
        初始化跟踪器
        Args:
            max_age: 轨迹最大未更新帧数（超过则标记为过期）
            max_cosine_distance: 余弦距离匹配阈值（越小匹配越严格）
            n_init: 轨迹确认所需的连续匹配帧数
            license_match_thresh: 车牌相似度匹配阈值（0-1，越大匹配越严格）
        """
        # 初始化原生 DeepSORT 跟踪器
        self.base_tracker = BaseDeepSort(
            max_age=max_age,
            max_cosine_distance=max_cosine_distance,
            n_init=n_init
        )
        # 车牌匹配阈值
        self.license_match_thresh = license_match_thresh
        # 存储轨迹的车牌历史（跨帧关联：track_id -> 最新车牌文本）
        self.track_licenses = {}
        # 帧计数（用于调试，与库内帧计数同步）
        self.frame_count = 0

    def _raw_dets_to_detections(self, raw_detections, frame, instance_masks=None, others=None):
        """
        提炼自 deep_sort_realtime 的 update_tracks 方法
        实现：raw_detections → 标准 Detection 对象（含格式校验、特征提取、NMS 过滤）
        完全复用库内原生逻辑，确保接口兼容
        Args:
            raw_detections: 库标准格式 List[ Tuple[ [left,top,w,h] , confidence, detection_class] ]
            frame: 输入帧 (np.ndarray, [H,W,C])
            instance_masks: 实例掩码（与库内一致，默认None）
            others: 自定义扩展信息列表（与 raw_detections 长度一致，默认None）
        Returns:
            List[Detection]: 库标准 Detection 对象列表（已完成 NMS 过滤）
        """
        detections = []

        # 步骤1：空值判断与格式校验
        if len(raw_detections) == 0:
            return detections
        if not self.base_tracker.polygon:
            # 校验 bbox 格式为 [left,top,w,h]，过滤宽/高为 0 的无效检测框
            assert len(raw_detections[0][0]) == 4, "bbox 格式必须为 [left,top,w,h]"
            raw_detections = [d for d in raw_detections if d[0][2] > 0 and d[0][3] > 0]
            if len(raw_detections) == 0:
                return detections

        # 步骤2：批量提取外观特征（复用库内 generate_embeds，效率最优）
        embeds = self.base_tracker.generate_embeds(
            frame, raw_detections, instance_masks=instance_masks
        )

        # 步骤3：构造库标准 Detection 对象（复用库内 create_detections）
        detections = self.base_tracker.create_detections(
            raw_detections, embeds, instance_masks=instance_masks, others=others
        )

        # 步骤4：NMS 非极大值抑制（去重重叠度高的检测框）
        if self.base_tracker.nms_max_overlap < 1.0:
            boxes = np.array([d.ltwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            nms_indices = self.base_tracker.non_max_suppression(
                boxes, self.base_tracker.nms_max_overlap, scores
            )
            detections = [detections[i] for i in nms_indices]

        return detections

    def _prepare_detections_with_license(self, vehicle_detections: List[Dict], frame: np.ndarray):
        """
        外部传入的车辆检测结果 → 构造带车牌的标准 Detection 对象
        核心：从输入字典中读取车牌信息，绑定到 Detection.others 字段
        Args:
            vehicle_detections: 外部传入格式 List[Dict]，包含 'bbox'/'conf'/'class_id'/'license'
            frame: 输入帧 (np.ndarray, [H,W,C])
        Returns:
            List[Detection]: 带车牌属性的标准 Detection 对象列表
            Dict[int, str]: Detection 索引 → 车牌信息映射（帧内局部使用）
        """
        # 步骤1：转换为 deep_sort_realtime 标准 raw_detections 格式，同时收集车牌
        raw_detections = []
        license_list = []  # 与 raw_detections 一一对应，存储车牌信息

        for det in vehicle_detections:
            # 解析输入字典中的字段
            bbox_xyxy = det['bbox']  # 外部传入 (x1,y1,x2,y2) 格式
            conf = det['conf']
            cls_id = det['class_id']
            license_text = det.get('license', "").strip()  # 读取外部已识别的车牌

            # 转换 bbox 格式：(x1,y1,x2,y2) → [left,top,w,h]（库要求格式）
            x1, y1, x2, y2 = bbox_xyxy
            ltwh = [x1, y1, x2 - x1, y2 - y1]

            # 构造库标准 raw_detections 元素
            raw_detections.append((ltwh, conf, cls_id))
            # 收集车牌信息（无车牌则为空字符串）
            license_list.append(license_text)

        # 步骤2：构造自定义扩展信息，绑定车牌到 Detection.others
        others = [{LICENSE_KEY: lic} for lic in license_list]
        detections = self._raw_dets_to_detections(raw_detections, frame, others=others)

        # 步骤3：构建 Detection 索引与车牌的映射（帧内局部使用，避免信息错乱）
        det_license_map = {}
        for det_idx, (detection, lic_text) in enumerate(zip(detections, license_list)):
            det_license_map[det_idx] = lic_text
            # 额外兜底：确保车牌信息已存入 Detection.others（避免库内逻辑清空该字段）
            if not hasattr(detection, 'others') or detection.others is None:
                detection.others = {}
            detection.others[LICENSE_KEY] = lic_text

        return detections, det_license_map

    def update_tracker(self, vehicle_detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        核心跟踪方法：接收带车牌的车辆检测结果，完成车牌优先匹配 + 原生 DeepSORT 跟踪
        Args:
            vehicle_detections: 外部传入 List[Dict]，包含 'bbox'(x1,y1,x2,y2)/'conf'/'class_id'/'license'
            frame: 输入帧 (np.ndarray, [H,W,C])
        Returns:
            List[Dict]: 包含 track_id/bbox/class_id/license 的最终轨迹结果
        """
        # ===== 新增：初始化总耗时计时 =====
        total_start = time.perf_counter()
        step_times = {}  # 记录各步骤耗时
        step_start = 0

        self.frame_count += 1
        logging.debug(f"Processing frame: {self.frame_count}")

        # 步骤1：构造带车牌的标准 Detection 对象（从外部读取车牌，无内部检测逻辑）
        step_start = time.perf_counter()
        detections, det_license_map = self._prepare_detections_with_license(vehicle_detections, frame)
        step_times["1. 构造Detection对象"] = time.perf_counter() - step_start

        if len(detections) == 0:
            # 无有效检测框，执行卡尔曼滤波预测后返回空结果
            step_start = time.perf_counter()
            self.base_tracker.tracker.predict()
            step_times["2. 卡尔曼滤波预测（空检测）"] = time.perf_counter() - step_start

            # 打印空检测场景的耗时
            total_time = time.perf_counter() - total_start
            self._print_tracker_step_times(step_times, total_time)
            return []

        # 步骤2：车牌优先匹配（带车牌的 Detection → 现有已确认轨迹）
        step_start = time.perf_counter()
        tracked_tracks = self.base_tracker.tracker.tracks  # 获取当前所有轨迹
        matched_det_indices = set()  # 记录已通过车牌匹配的 Detection 索引
        license_matched_tracks = {}  # 记录：track_id → Detection 索引

        for det_idx, (detection, license_text) in enumerate(zip(detections, det_license_map.values())):
            if not license_text:
                continue  # 无车牌的 Detection，后续走原生 DeepSORT 匹配

            # 查找最优车牌匹配的轨迹
            best_track = None
            max_sim = 0.0

            for track in tracked_tracks:
                # 跳过未确认/长时间未更新的轨迹（与库内逻辑一致，保证跟踪稳定性）
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                track_id = track.track_id
                prev_license = self.track_licenses.get(track_id, "")
                if not prev_license:
                    continue  # 轨迹无历史车牌，无法进行车牌匹配

                # 计算车牌文本相似度（Levenshtein 距离，抗字符错位/缺失）
                sim = ratio(license_text, prev_license)
                if sim > max_sim and sim >= self.license_match_thresh:
                    max_sim = sim
                    best_track = track

            if best_track:
                # 车牌匹配成功：更新轨迹（复用库内 Track 原生 update 方法）
                best_track.time_since_update = 0  # 重置未更新帧数，避免轨迹过期
                # best_track.state = Track.State.Tracked  # 标记为活跃跟踪状态 这行似乎不需要
                best_track.update(self.base_tracker.tracker.kf, detection)  # 传入标准 Detection 对象，兼容库内接口

                # 记录匹配关系，更新轨迹车牌历史
                matched_det_indices.add(det_idx)
                license_matched_tracks[best_track.track_id] = det_idx
                self.track_licenses[best_track.track_id] = license_text
        step_times["2. 车牌优先匹配（核心）"] = time.perf_counter() - step_start

        # 步骤3：剩余未匹配的 Detection → 复用 DeepSORT 原生跟踪逻辑
        step_start = time.perf_counter()
        remaining_detections = [
            detections[det_idx] for det_idx in range(len(detections))
            if det_idx not in matched_det_indices
        ]
        step_times["3. 筛选剩余未匹配Detection"] = time.perf_counter() - step_start

        # 库原生跟踪流程：卡尔曼滤波预测 → 匈牙利算法匹配更新
        step_start = time.perf_counter()
        self.base_tracker.tracker.predict()
        self.base_tracker.tracker.update(remaining_detections, today=None)
        step_times["4. 原生DeepSORT（预测+更新）"] = time.perf_counter() - step_start

        # 步骤4：整合轨迹结果，补充车牌信息，构造外部返回格式
        step_start = time.perf_counter()
        result_tracks = []
        all_tracks = self.base_tracker.tracker.tracks  # 获取所有更新后的轨迹

        for track in all_tracks:
            if not track.is_confirmed():
                continue  # 跳过未确认的轨迹（过滤噪声轨迹）

            track_id = track.track_id
            # 转换 bbox 格式：[left,top,w,h] → (x1,y1,x2,y2)（与外部输入格式一致）
            ltwh = track.to_ltwh()
            x1, y1, w, h = ltwh
            bbox_xyxy = [x1, y1, x1 + w, y1 + h]
            class_id = track.get_det_class()
            license_text = ""

            # 优先级1：从车牌匹配结果中获取（最准确）
            if track_id in license_matched_tracks:
                det_idx = license_matched_tracks[track_id]
                license_text = det_license_map.get(det_idx, "")
            # 优先级2：从原生匹配的 Detection 对象中提取（无车牌匹配时）
            elif track.get_det_supplementary() is not None:
                license_text = track.get_det_supplementary().get(LICENSE_KEY, "")
            # 优先级3：从轨迹历史车牌中补充（车辆暂时未识别到车牌时）
            if not license_text:
                license_text = self.track_licenses.get(track_id, "")
            else:
                # 更新轨迹历史车牌（保持最新车牌信息）
                self.track_licenses[track_id] = license_text

            # 构造最终返回结果
            result_tracks.append({
                'track_id': track_id,
                'bbox': bbox_xyxy,
                'class_id': class_id,
                'license': license_text.strip()
            })
        step_times["5. 整合轨迹结果（补车牌）"] = time.perf_counter() - step_start

        # ===== 新增：打印所有步骤耗时 =====
        total_time = time.perf_counter() - total_start
        self._print_tracker_step_times(step_times, total_time)

        return result_tracks

    # ===== 新增：耗时打印辅助方法 =====
    def _print_tracker_step_times(self, step_times: Dict[str, float], total_time: float):
        """
        打印跟踪器各步骤耗时及占比
        Args:
            step_times: 步骤名称 → 耗时（秒）
            total_time: 跟踪器总耗时（秒）
        """
        logging.info(f"\n===== 跟踪器（update_tracker）耗时统计（帧{self.frame_count}） =====")
        logging.info(f"跟踪器总耗时：{total_time:.4f} 秒")
        for step_name, step_time in step_times.items():
            ratio = (step_time / total_time) * 100 if total_time > 0.0001 else 0.0
            logging.info(f"{step_name}：{step_time:.4f} 秒（占比：{ratio:.2f}%）")
        logging.info("=" * 50)
