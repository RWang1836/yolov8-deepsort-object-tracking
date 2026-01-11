import os
import cv2
import argparse
import logging
import numpy as np
import sys
import time
from pathlib import Path

from detections import Detector
from tracking_with_plate import Tracker  # 对应无valid_indices、仅接收带车牌Detection的版本
from utils_with_plate import Visualizer
# 获取yolov8_plate目录的绝对路径
YOLOV8_PLATE_DIR = Path(__file__).parent / "yolov8_plate"
# 将该目录加入sys.path，让Python能找到ultralytics_plate模块
sys.path.insert(0, str(YOLOV8_PLATE_DIR))
from yolov8_plate.detect_rec_plate import PlateDetector  # 车牌检测识别封装

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
VEHICLE_CLASS_IDS = [2, 5, 7]  # 汽车、公交车、卡车
PLATE_CONF_THRESH = 0.5
LICENSE_MATCH_THRESH = 0.8  # 车牌字符相似度阈值

# 新增：标定框绘制配置（可根据需求调整）
CALIBRATION_BOX_COLOR = (0, 255, 255)  # 标定框颜色：黄色
CALIBRATION_BOX_THICKNESS = 2  # 标定框线宽
CALIBRATION_BOX_LABEL = "License Valid Area"  # 标定框标签


class LicenseTracker:
    def __init__(self, vehicle_model, plate_det_model, plate_rec_model):
        # 初始化车辆检测器
        self.vehicle_detector = Detector(vehicle_model)
        # 初始化车牌检测识别器（核心：所有车牌逻辑集中在此）
        self.plate_detector = PlateDetector(
            detect_model=plate_det_model,
            rec_model=plate_rec_model
        )
        # 初始化跟踪器（仅负责跟踪，无需处理车牌检测）
        self.tracker = Tracker(license_match_thresh=LICENSE_MATCH_THRESH)
        # 初始化可视化工具
        self.visualizer = Visualizer(VEHICLE_CLASS_IDS, self.vehicle_detector.class_names)
        # 标定区域（示例：画面中间区域，需根据实际场景调整）
        self.calibration_area = None  # 格式: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        # ===== 新增：性能统计相关初始化 =====
        self.frame_start_time = 0  # 单帧开始时间
        self.frame_end_time = 0  # 单帧结束时间
        self.total_process_time = 0.0  # 所有帧总处理时间
        self.processed_frame_count = 0  # 已处理帧数
        self.step_times = {}  # 关键步骤耗时记录（车辆检测/车牌识别/跟踪/可视化）

    def set_calibration_area(self, area):
        """设置车牌识别有效区域"""
        self.calibration_area = np.array(area, dtype=np.int32)

    def is_inside_calibration(self, bbox):
        """判断车牌框是否在标定区域内（bbox格式：(x1,y1,x2,y2)）"""
        if self.calibration_area is None:
            return True  # 无标定区域时默认全部有效
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) // 2, (y1 + y2) // 2)  # 取车牌中心判断
        return cv2.pointPolygonTest(self.calibration_area, center, False) >= 0

        # 新增：绘制标定区域框
    def _draw_calibration_box(self, frame):
        """在画面上绘制标定区域框（仅当标定区域已设置时绘制）"""
        if self.calibration_area is None:
            return frame

        # 1. 绘制标定区域多边形（四边形）
        cv2.polylines(
            frame,
            [self.calibration_area],  # 多边形顶点坐标
            isClosed=True,  # 闭合多边形（首尾相连）
            color=CALIBRATION_BOX_COLOR,
            thickness=CALIBRATION_BOX_THICKNESS
        )

        # 2. 绘制标定区域标签（在区域左上角，提升可读性）
        # 获取标定区域左上角坐标（取第一个顶点作为参考）
        label_x, label_y = self.calibration_area[0]
        # 调整标签位置，避免与框线重叠
        label_y = max(20, label_y - 10)

        # 绘制标签背景（可选，提升文字可读性）
        (label_w, label_h), _ = cv2.getTextSize(
            CALIBRATION_BOX_LABEL,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )
        cv2.rectangle(
            frame,
            (label_x, label_y - label_h - 5),
            (label_x + label_w + 5, label_y + 5),
            (255, 255, 255),  # 白色背景
            -1  # 填充背景
        )

        # 绘制标签文字
        cv2.putText(
            frame,
            CALIBRATION_BOX_LABEL,
            (label_x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            CALIBRATION_BOX_COLOR,
            1
        )

        return frame

    def process_frame(self, frame):
        """单帧处理：车辆检测 → 车牌识别 → 轨迹更新 → 可视化"""
        # ===== 新增：记录单帧开始时间 =====
        self.frame_start_time = time.perf_counter()  # 高精度计时
        self.step_times = {}  # 重置当前帧步骤耗时
        step_start = 0

        # 1. 车辆检测（获取原始车辆检测结果）
        step_start = time.perf_counter()
        vehicle_detections = self.vehicle_detector.detect(
            frame, VEHICLE_CLASS_IDS, conf_thresh=0.4
        )
        self.step_times["车辆检测"] = time.perf_counter() - step_start

        # 2. 车牌检测与识别（核心：所有车牌逻辑集中在此，构造供跟踪器使用的格式）
        step_start = time.perf_counter()
        tracked_input_detections = []  # 供Tracker.update_tracker使用的输入格式
        for det_tuple in vehicle_detections:
            bbox_ltwh, conf, cls_id = det_tuple  # 原始检测：(x1,y1,w,h)
            x1, y1, w, h = bbox_ltwh
            x2, y2 = x1 + w, y1 + h  # 转换为xyxy格式，用于截取ROI
            bbox_xyxy = (x1, y1, x2, y2)

            # 初始化车牌信息
            license_text = ""

            # 2.1 截取车辆ROI（避免超出帧边界）
            h_frame, w_frame = frame.shape[:2]
            x1_clip = max(0, int(x1))
            y1_clip = max(0, int(y1))
            x2_clip = min(w_frame, int(x2))
            y2_clip = min(h_frame, int(y2))
            vehicle_roi = frame[y1_clip:y2_clip, x1_clip:x2_clip]

            # 2.2 车牌检测（仅处理有效ROI）
            if vehicle_roi.size > 0:  # 确保ROI非空
                plate_bboxes = self.plate_detector.detect_plate(vehicle_roi)
                for plate_bbox in plate_bboxes:
                    # 解析车牌检测结果（p_x1,p_y1,p_x2,p_y2为ROI内坐标）
                    p_x1, p_y1, p_x2, p_y2, p_conf, p_type = plate_bbox

                    # 过滤低置信度假车牌
                    if p_conf < PLATE_CONF_THRESH:
                        continue

                    # 2.3 转换为原图坐标（ROI坐标 → 帧全局坐标）
                    orig_p_x1 = p_x1 + x1_clip
                    orig_p_y1 = p_y1 + y1_clip
                    orig_p_x2 = p_x2 + x1_clip
                    orig_p_y2 = p_y2 + y1_clip
                    plate_bbox_original = (orig_p_x1, orig_p_y1, orig_p_x2, orig_p_y2)

                    # 2.4 检查是否在标定区域内
                    if self.is_inside_calibration(plate_bbox_original):
                        # 2.5 车牌识别（获取车牌文本）
                        plate_text, plate_color = self.plate_detector.recognize_plate(vehicle_roi, plate_bbox)
                        license_text = plate_text.strip()
                        break  # 只取第一个有效车牌，提升效率

            # 2.6 构造跟踪器所需输入格式（与Tracker.update_tracker兼容）
            tracked_input_detections.append({
                'bbox': bbox_xyxy,  # (x1,y1,x2,y2)，供跟踪器转换ltwh
                'conf': conf,  # 车辆检测置信度
                'class_id': cls_id,  # 车辆类别ID
                'license': license_text  # 已识别的车牌（无则为空字符串）
            })
        self.step_times["车牌检测与识别"] = time.perf_counter() - step_start

        # 3. 更新轨迹（传入带车牌的车辆检测结果，跟踪器仅负责匹配和跟踪）
        # 修正：原代码传入vehicle_detections（原始元组），现传入构造好的tracked_input_detections
        step_start = time.perf_counter()
        tracks = self.tracker.update_tracker(tracked_input_detections, frame)
        self.step_times["轨迹更新（DeepSORT）"] = time.perf_counter() - step_start

        # 4. 可视化（轨迹、车辆计数）
        step_start = time.perf_counter()
        self.visualizer.draw_tracks(frame, tracks)
        self.visualizer.draw_counts(frame)
        self._draw_calibration_box(frame)
        self.step_times["结果可视化"] = time.perf_counter() - step_start

        # ===== 单帧性能统计收尾 =====
        self.frame_end_time = time.perf_counter()
        frame_total_time = self.frame_end_time - self.frame_start_time  # 单帧总耗时
        self.total_process_time += frame_total_time  # 累加总处理时间
        self.processed_frame_count += 1  # 累加已处理帧数

        # ===== 仅终端打印性能信息（删除画面可视化FPS） =====
        self._print_performance_info(frame_total_time)

        return frame, tracks

    def _print_performance_info(self, frame_total_time):
        """仅在终端打印当前帧性能信息（关键步骤耗时、瞬时FPS、平均FPS）"""
        # 计算FPS
        instant_fps = 1.0 / frame_total_time if frame_total_time > 0 else 0.0  # 瞬时FPS
        avg_fps = self.processed_frame_count / self.total_process_time if self.total_process_time > 0 else 0.0  # 平均FPS

        # 打印性能日志（格式化输出，更易阅读）
        logging.info(f"===== 帧 {self.processed_frame_count} 性能统计 =====")
        logging.info(f"单帧总耗时：{frame_total_time:.4f} 秒")
        logging.info(f"瞬时FPS：{instant_fps:.2f} FPS | 平均FPS：{avg_fps:.2f} FPS")
        for step_name, step_time in self.step_times.items():
            step_ratio = (step_time / frame_total_time) * 100 if frame_total_time > 0 else 0.0
            logging.info(f"{step_name}：{step_time:.4f} 秒（占比：{step_ratio:.2f}%）")
        logging.info("=" * 45 + "\n")

    def get_final_performance_summary(self):
        """返回最终性能汇总（视频处理完毕后在终端打印）"""
        if self.processed_frame_count == 0:
            return "未处理任何帧，无性能汇总信息"

        avg_frame_time = self.total_process_time / self.processed_frame_count
        avg_fps = self.processed_frame_count / self.total_process_time

        summary = f"""
            ===== 最终性能汇总 =====
            总处理帧数：{self.processed_frame_count} 帧
            总处理时间：{self.total_process_time:.2f} 秒
            平均单帧耗时：{avg_frame_time:.4f} 秒
            平均FPS：{avg_fps:.2f} FPS
            =======================
        """
        return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Tracking with License Recognition")
    parser.add_argument("--input", type=str, default="data/traffic_night.mp4")
    parser.add_argument("--output", type=str, default="output/traffic_with_plate.mp4")
    parser.add_argument("--vehicle_model", type=str, default="Yolo-Weights/yolov8s.pt")
    parser.add_argument("--plate_det_model", type=str, default="yolov8_plate/weights/yolov8s.pt")
    parser.add_argument("--plate_rec_model", type=str, default="yolov8_plate/weights/plate_rec_color.pth")
    return parser.parse_args()


def main():
    args = parse_args()

    # 检查输出目录是否存在，不存在则创建
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 初始化LicenseTracker（所有逻辑入口）
    license_tracker = LicenseTracker(
        args.vehicle_model,
        args.plate_det_model,
        args.plate_rec_model
    )
    # 设置标定区域（示例：画面下方中间区域，可根据实际场景调整）
    license_tracker.set_calibration_area([(300, 400), (900, 400), (900, 600), (300, 600)])

    # 视频读取与写入配置
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logging.error(f"无法打开输入视频：{args.input}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # 兼容无FPS信息的视频
    out = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    if not out.isOpened():
        logging.error(f"无法创建输出视频：{args.output}")
        cap.release()
        return

    logging.info("开始处理视频...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频处理完毕

        # 单帧处理
        processed_frame, _ = license_tracker.process_frame(frame)

        # 写入输出视频并显示
        out.write(processed_frame)
        cv2.imshow("Tracking with License Recognition", processed_frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("用户手动退出视频处理")
            break

    # ===== 终端打印最终性能汇总 =====
    final_summary = license_tracker.get_final_performance_summary()
    logging.info(final_summary)

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info(f"视频处理完成，结果保存至：{args.output}")


if __name__ == "__main__":
    main()