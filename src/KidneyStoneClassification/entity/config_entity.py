from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path 
    source_URL: str
    local_data_file: Path 
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: int
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes:int

@dataclass(frozen=True)
class PreprocessingConfig:
    root_dir: Path
    processed_data_path: Path
    params_clahe_clip: float
    params_denoise_strength: float
    params_intensity_rescale: bool