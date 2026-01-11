# ultralytics_plate YOLO 🚀, AGPL-3.0 license

__version__ = "8.1.2"

from ultralytics_plate.data.explorer.explorer import Explorer
from ultralytics_plate.models import RTDETR, SAM, YOLO
from ultralytics_plate.models.fastsam import FastSAM
from ultralytics_plate.models.nas import NAS
from ultralytics_plate.utils import SETTINGS as settings
from ultralytics_plate.utils.checks import check_yolo as checks
from ultralytics_plate.utils.downloads import download

__all__ = "__version__", "YOLO", "NAS", "SAM", "FastSAM", "RTDETR", "checks", "download", "settings", "Explorer"
