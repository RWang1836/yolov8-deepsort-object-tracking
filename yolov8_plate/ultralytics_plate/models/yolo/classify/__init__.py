# ultralytics_plate YOLO 🚀, AGPL-3.0 license

from ultralytics_plate.models.yolo.classify.predict import ClassificationPredictor
from ultralytics_plate.models.yolo.classify.train import ClassificationTrainer
from ultralytics_plate.models.yolo.classify.val import ClassificationValidator

__all__ = "ClassificationPredictor", "ClassificationTrainer", "ClassificationValidator"
