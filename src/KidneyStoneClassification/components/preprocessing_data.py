import cv2
import os
from KidneyStoneClassification import logger
import tqdm
import numpy as np
import torchio as tio
from pathlib import Path
from KidneyStoneClassification.entity.config_entity import PreprocessingConfig

class Preprocessing:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def apply_clahe(self, image):
        clahe = cv2.createCLAHE(clipLimit=self.config.params_clahe_clip, tileGridSize=(8,8))
        return clahe.apply(image)

    def denoise(self, image):
        return cv2.fastNlMeansDenoising(image, None, self.config.params_denoise_strength, 7, 21)

    def intensity_rescale(self, image):
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    
    def process_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Error: Input image not found at {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Error: Failed to load image. Ensure {image_path} is a valid grayscale image.")

        image = self.apply_clahe(image)
        image = self.denoise(image)
        if self.config.params_intensity_rescale:
            image = self.intensity_rescale(image)
        return image
    
    def save_processed_image(self, image, output_path: str):
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        cv2.imwrite(output_path, image)
        if not os.path.exists(output_path):
            logger.error(f"Error: Image was not saved to {output_path}")
            
    def process_directory(self, input_dir, output_dir):
        """
        Process all images in a directory structure, maintaining the same organization
        
        Args:
            input_dir: Root directory containing class folders
            output_dir: Output directory where processed images will be saved
        """
        logger.info(f"Starting to process images from {input_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all class folders
        class_folders = [d for d in os.listdir(input_dir) 
                         if os.path.isdir(os.path.join(input_dir, d))]
        
        total_processed = 0
        total_errors = 0
        
        for class_name in class_folders:
            logger.info(f"Processing class: {class_name}")
            class_path = os.path.join(input_dir, class_name)
            output_class_path = os.path.join(output_dir, class_name)
            os.makedirs(output_class_path, exist_ok=True)
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
                try:
                    input_path = os.path.join(class_path, img_file)
                    output_path = os.path.join(output_class_path, img_file)
                    
                    # Process and save the image
                    processed_img = self.process_image(input_path)
                    self.save_processed_image(processed_img, output_path)
                    total_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {str(e)}")
                    total_errors += 1
        
        logger.info(f"Processing complete! Processed {total_processed} images with {total_errors} errors")
        return total_processed, total_errors