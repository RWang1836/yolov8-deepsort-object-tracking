"""
tracking_with_plate.py (修改版，关联全局plate_tracker)
----------------------
主脚本：集成车牌识别的车辆跟踪流程
"""
import os
import cv2
import argparse
import logging
import torch
from detections import Detector
from tracking import PlateTracker  # 导入修改后的跟踪器
from utils import Visualizer
from yolov8_plate_master.detect_rec_plate import load_model as load_plate_detector, \
                                                  init_model as init_plate_recognizer, \
                                                  det_rec_plate

# 声明全局plate_tracker，供DeepSORT的gated_metric调用
global plate_tracker
plate_tracker = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
class_ids = [2, 3, 5, 7]  # 仅跟踪车辆类：car, motorcycle, bus, truck

def parse_args():
    parser = argparse.ArgumentParser(description="带车牌识别的车辆跟踪 (YOLOv8 + DeepSORT)")
    parser.add_argument("--input", type=str, default="data/crosswalk_traffic.mp4", help="输入视频路径或0(摄像头)")
    parser.add_argument("--output", type=str, default="output/plate_tracking_output.mp4", help="输出视频路径")
    parser.add_argument("--yolo_model", type=str, default="Yolo-Weights/yolov8m.pt", help="YOLOv8模型路径")
    parser.add_argument("--plate_det_model", type=str, default="yolov8-plate-master/weights/yolov8s.pt", help="车牌检测模型")
    parser.add_argument("--plate_rec_model", type=str, default="yolov8-plate-master/weights/plate_rec_color.pth", help="车牌识别模型")
    parser.add_argument("--frame_width", type=int, default=640, help="显示宽度(0=不缩放)")
    parser.add_argument("--conf_thresh", type=float, default=0.4, help="置信度阈值")
    return parser.parse_args()

def main():
    global plate_tracker
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用设备: {device}")

    # 初始化车辆检测器
    logging.info("加载YOLOv8车辆检测器...")
    detector = Detector(args.yolo_model)

    # 初始化车牌识别模型
    logging.info("加载车牌检测与识别模型...")
    plate_det_model = load_plate_detector(args.plate_det_model, device)
    plate_rec_model = init_plate_recognizer(device, args.plate_rec_model, is_color=True)

    # 初始化带车牌特征的跟踪器（赋值给全局变量）
    plate_tracker = PlateTracker()

    # 初始化可视化工具
    visualizer = Visualizer(class_ids, detector.class_names)

    # 打开视频流
    cap = cv2.VideoCapture(0 if args.input == "0" else args.input)
    if not cap.isOpened():
        logging.error(f"无法打开输入: {args.input}")
        return

    # 视频参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 车辆检测
        detections = detector.detect(frame, class_ids, args.conf_thresh)

        # 2. 对检测到的车辆进行车牌识别
        plate_results = []
        for det in detections:
            # 适配detector.detect的输出格式（x1, y1, w, h, conf, cls）
            x1, y1, w, h, conf, cls = det
            x2, y2 = x1 + w, y1 + h
            vehicle_roi = frame[y1:y2, x1:x2]  # 裁剪车辆区域
            if vehicle_roi.size == 0:
                continue

            # 车牌识别 (返回格式: [{'plate_no': '京A12345', 'plate_color': 'blue', ...}])
            plates = det_rec_plate(vehicle_roi, vehicle_roi, plate_det_model, plate_rec_model)
            if plates:
                plate_results.append({
                    "vehicle_bbox": (x1, y1, x2, y2),  # 车辆边界框
                    "plate_info": plates[0]            # 车牌信息
                })

        # 3. 关联车牌信息到跟踪器
        plate_tracker._associate_plate_to_tracks(plate_tracker.tracker.tracks)

        # 4. 融合车牌特征的跟踪更新
        tracks = plate_tracker.update_tracker(detections, frame, plate_results)

        # 5. 可视化 (绘制跟踪框、ID、车牌信息)
        visualizer.draw_tracks(frame, tracks, args.conf_thresh)
        visualizer.draw_plates(frame, tracks, plate_tracker.track_plate_info)  # 绘制车牌
        visualizer.draw_counts(frame)
        visualizer.draw_fps(frame)

        # 保存与显示
        out.write(frame)
        if args.frame_width > 0:
            scale = args.frame_width / width
            frame = cv2.resize(frame, (args.frame_width, int(height * scale)))
        cv2.imshow("车辆跟踪(带车牌识别)", frame)

        if cv2.waitKey(1) in [ord('q'), 27]:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logging.info(f"处理完成，输出保存至: {args.output}")

if __name__ == "__main__":
    main()