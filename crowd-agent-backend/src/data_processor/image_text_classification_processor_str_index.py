import logging
import csv
from typing import Optional
from dataclasses import dataclass
from src.data_processor.data_processor import DataProcessor


logger = logging.getLogger(__name__)


@dataclass
class ImageTextClassificationInputExample:

    index: str
    contents: str
    label: Optional[int]
    image_path: str


class ImageTextClassificationProcessor(DataProcessor):

    def get_train_examples(self, data_dir, skip_head=False) -> list[ImageTextClassificationInputExample]:
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        train = self._read_csv(data_dir, skip_head)
        return self._create_examples(train)

    def _read_csv(self, data_dir, skip_head=False):
        corpus = []
        with open(data_dir, 'r') as f:
            reader = csv.reader(f)
            if skip_head:
                next(reader)
            for line in reader:
                corpus.append([line[0], int(line[1]), line[2], line[3]])
        return corpus

    def _create_examples(self, corpus) -> list[ImageTextClassificationInputExample]:
        examples = []
        for (i, example) in enumerate(corpus):
            index = example[0]
            label = example[1]
            text = example[2]
            image_path = example[3]
            examples.append(ImageTextClassificationInputExample(index=index, contents=text, label=label, image_path=image_path))
        return examples
