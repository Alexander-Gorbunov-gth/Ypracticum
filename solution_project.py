from pathlib import Path
from logging import getLogger

from src.services.data_utils import TextDataPreparation
from src.domain.constans import DataConstants
from transformers import BertTokenizerFast

from src.services.tokens import TokensPreparation

logger = getLogger(__name__)
BASE_DIR = Path().resolve()

raw_data_file = Path(BASE_DIR / DataConstants.raw_data_patch)
dataset_processed_file = Path(BASE_DIR / DataConstants.dataset_processed_patch)


tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


text_service = TextDataPreparation(
    tokens_service=TokensPreparation,
)

text_service_result = text_service.create_process_dataset(
    input_path=raw_data_file,
    output_path=dataset_processed_file,
    batch_size=DataConstants.batch_size,
    min_text_length=4,
)
logger.info(text_service_result)
