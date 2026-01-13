import gzip
import json
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Script started")

try:
    with gzip.open('simplified_qa/v1.0-simplified_simplified-nq-train.jsonl.gz', 'rt', encoding='utf-8') as f:
        logging.info("File opened successfully")
        first_line = f.readline()
        logging.info("First line read")
        data = json.loads(first_line)
        logging.info(f"Question from first line: {data.get('question_text', 'Not found')}")
except Exception as e:
    logging.error(f"Error occurred: {str(e)}")

logging.info("Script finished")
