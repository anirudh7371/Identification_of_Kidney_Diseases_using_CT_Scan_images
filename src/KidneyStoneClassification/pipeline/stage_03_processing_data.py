from KidneyStoneClassification.config.configuration import ConfigurationManager
from KidneyStoneClassification.components.prepare_base_model import PrepareBaseModel
from KidneyStoneClassification import logger
from KidneyStoneClassification.entity.config_entity import PreprocessingConfig
import cv2
import os
from tqdm import tqdm
from pathlib import Path


STAGE_NAME = "Processing Data"

class PreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        preprocessing_config = config.get_preprocessing_config()
        preprocessing = Preprocessing(config=preprocessing_config)

        # Define input and output directories
        input_dir = preprocessing_config.root_dir
        output_dir = preprocessing_config.processed_data_path

        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")

        # Process dataset
        preprocessing.process_directory(input_dir, output_dir)


class Preprocessing:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def apply_clahe(self, image):
        clahe = cv2.createCLAHE(clipLimit=self.config.params_clahe_clip, tileGridSize=(8, 8))
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
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(output_path, image)
        if not os.path.exists(output_path):
            logger.error(f"Error: Image was not saved to {output_path}")
    
    def process_directory(self, input_dir, output_dir):
        logger.info(f"Starting to process images from {input_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert Path object to string if needed
        input_dir_str = str(input_dir) if isinstance(input_dir, Path) else input_dir
        output_dir_str = str(output_dir) if isinstance(output_dir, Path) else output_dir
        
        # Check if input directory exists
        if not os.path.exists(input_dir_str):
            logger.error(f"Input directory does not exist: {input_dir_str}")
            return 0, 0
        
        # List contents of input directory for debugging
        logger.info(f"Contents of input directory: {os.listdir(input_dir_str)}")
        
        # Find class folders
        class_folders = [d for d in os.listdir(input_dir_str) 
                         if os.path.isdir(os.path.join(input_dir_str, d))]
        
        if not class_folders:
            logger.error(f"No class folders found in {input_dir_str}")
            return 0, 0
            
        logger.info(f"Found class folders: {class_folders}")
        
        total_processed = 0
        total_errors = 0
        
        for class_name in class_folders:
            class_path = os.path.join(input_dir_str, class_name)
            output_class_path = os.path.join(output_dir_str, class_name)
            os.makedirs(output_class_path, exist_ok=True)
            
            logger.info(f"Processing class: {class_name} from {class_path}")
            
            
            # Find all image files with correct extension check
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            logger.info(f"Found {len(image_files)} images in {class_name}")
            
            for img_file in tqdm(image_files, desc=f"Processing {class_name}"):
                try:
                    input_path = os.path.join(class_path, img_file)
                    output_path = os.path.join(output_class_path, img_file)
                    
                    processed_img = self.process_image(input_path)
                    self.save_processed_image(processed_img, output_path)
                    total_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {str(e)}")
                    total_errors += 1
        
        logger.info(f"Processing complete! Processed {total_processed} images with {total_errors} errors")
        return total_processed, total_errors

if __name__ == "__main__":
    try:
        logger.info(f"\n********************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        obj = PreprocessingPipeline()
        obj.main()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e