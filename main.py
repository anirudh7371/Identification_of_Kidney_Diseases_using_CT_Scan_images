from KidneyStoneClassification import logger 
from KidneyStoneClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from KidneyStoneClassification.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from KidneyStoneClassification.pipeline.stage_03_processing_data import PreprocessingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>> Stage {STAGE_NAME} started. <<<<")
    obj = DataIngestionTrainingPipeline();
    obj.main()
    logger.info(f">>>> Stage {STAGE_NAME} completed. <<<<\n\n x======x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare base model"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
   prepare_base_model = PrepareBaseModelTrainingPipeline()
   prepare_base_model.main()
   logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Preprocessing the Data"
try:
     logger.info(f"***********")
     logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
     preprocessing_data = PreprocessingPipeline()
     preprocessing_data.main()
     logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e