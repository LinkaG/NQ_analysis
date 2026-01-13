import json
import gzip
import logging
from datetime import datetime
import os
import argparse
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing_report.txt'),
        logging.StreamHandler()
    ]
)

def normalize_question(text: str) -> str:
    """Нормализация текста вопроса"""
    return text.lower().strip()

def create_minimal_index(filepath: str) -> dict:
    """
    Создает минимальный индекс: вопрос -> {url, example_id, byte_offset}
    Не хранит тексты документов и другие большие поля
    """
    questions_index = {}
    processed = 0
    current_pos = 0
    
    logging.info(f"Creating index for {filepath}")
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            pbar = tqdm(desc="Indexing simplified NQ", unit=" questions")
            
            while True:
                try:
                    # Запоминаем текущую позицию
                    pos = f.tell()
                    line = f.readline()
                    if not line:
                        break
                        
                    data = json.loads(line.strip())
                    question = data.get('question_text', '').strip()
                    
                    if question:
                        normalized_question = normalize_question(question)
                        # Сохраняем только минимально необходимые данные и позицию в файле
                        questions_index[normalized_question] = {
                            'document_url': data.get('document_url', ''),
                            'example_id': data.get('example_id', ''),
                            'byte_offset': pos
                        }
                    
                    processed += 1
                    if processed % 1000 == 0:
                        pbar.update(1000)
                        pbar.set_postfix({'indexed': len(questions_index)})
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logging.error(f"Error processing line: {str(e)}")
                    continue
            
            pbar.close()
            
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        return questions_index
    
    logging.info(f"Finished indexing. Processed {processed} lines, indexed {len(questions_index)} questions")
    return questions_index

def get_full_data(filepath: str, byte_offset: int) -> dict:
    """Читает полные данные из определенной позиции в файле"""
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            f.seek(byte_offset)
            line = f.readline()
            return json.loads(line.strip())
    except Exception as e:
        logging.error(f"Error reading data at offset {byte_offset}: {str(e)}")
        return {}

def process_nq_open(input_file: str, output_file: str, unmatched_file: str, 
                   questions_index: dict, simplified_nq_path: str):
    """Обрабатывает файл из NQ-open датасета"""
    processed = matches = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout, \
             open(unmatched_file, 'w', encoding='utf-8') as funmatched:
            
            pbar = tqdm(desc=f"Processing {os.path.basename(input_file)}", unit=" questions")
            
            for line in fin:
                try:
                    data = json.loads(line.strip())
                    question = data.get('question', '').strip()
                    answer = data.get('answer', [])
                    normalized_question = normalize_question(question)
                    
                    if normalized_question in questions_index:
                        # Получаем индексную информацию
                        index_data = questions_index[normalized_question]
                        
                        # Читаем полные данные только если нашли совпадение
                        simplified_data = get_full_data(simplified_nq_path, index_data['byte_offset'])
                        
                        if simplified_data:
                            # Создаем новую запись
                            merged_data = {
                                'question': question,
                                'answer': answer,
                                'document_text': simplified_data.get('document_text', ''),
                                'document_url': index_data['document_url'],
                                'annotations': simplified_data.get('annotations', []),
                                'long_answer_candidates': simplified_data.get('long_answer_candidates', []),
                                'example_id': index_data['example_id']
                            }
                            
                            fout.write(json.dumps(merged_data, ensure_ascii=False) + '\n')
                            matches += 1
                    else:
                        funmatched.write(json.dumps({
                            'question': question,
                            'answer': answer
                        }, ensure_ascii=False) + '\n')
                    
                    processed += 1
                    if processed % 100 == 0:
                        pbar.update(100)
                        pbar.set_postfix({
                            'matches': f"{matches}/{processed}",
                            'rate': f"{matches/processed*100:.1f}%"
                        })
                
                except Exception as e:
                    logging.error(f"Error processing question: {str(e)}")
                    continue
            
            pbar.close()
    
    except Exception as e:
        logging.error(f"Error processing file {input_file}: {str(e)}")
    
    return processed, matches

def main():
    parser = argparse.ArgumentParser(description='Process NQ-open dataset')
    parser.add_argument('--dataset', choices=['train', 'dev', 'both'], default='both',
                      help='Which dataset to process')
    args = parser.parse_args()
    
    start_time = datetime.now()
    logging.info("Starting dataset processing")
    
    # Пути к файлам
    simplified_nq = 'simplified_qa/v1.0-simplified_simplified-nq-train.jsonl.gz'
    nq_open_train = 'nq_open/NQ-open.train.jsonl'
    nq_open_dev = 'nq_open/NQ-open.dev.jsonl'
    
    # Выходные файлы
    train_output = 'nq_open/NQ-open.train.merged.jsonl'
    dev_output = 'nq_open/NQ-open.dev.merged.jsonl'
    train_unmatched = 'nq_open/unmatched_questions_train.jsonl'
    dev_unmatched = 'nq_open/unmatched_questions_dev.jsonl'
    
    # Создаем индекс simplified NQ
    logging.info("Creating index for simplified NQ dataset...")
    questions_index = create_minimal_index(simplified_nq)
    if not questions_index:
        logging.error("Failed to create index for simplified NQ dataset")
        return
    
    logging.info(f"Created index for {len(questions_index)} questions")
    
    # Обрабатываем датасеты
    if args.dataset in ['train', 'both']:
        logging.info("Processing train dataset...")
        train_processed, train_matches = process_nq_open(
            nq_open_train, train_output, train_unmatched, 
            questions_index, simplified_nq
        )
        logging.info(f"Train dataset: processed {train_processed}, matched {train_matches}")
    
    if args.dataset in ['dev', 'both']:
        logging.info("Processing dev dataset...")
        dev_processed, dev_matches = process_nq_open(
            nq_open_dev, dev_output, dev_unmatched,
            questions_index, simplified_nq
        )
        logging.info(f"Dev dataset: processed {dev_processed}, matched {dev_matches}")
    
    end_time = datetime.now()
    logging.info(f"Processing completed in {end_time - start_time}")

if __name__ == "__main__":
    main()
