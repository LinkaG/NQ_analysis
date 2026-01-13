import json
import gzip
import logging
from datetime import datetime
import os
import argparse
from tqdm import tqdm
import gc

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

def process_chunk(filepath: str, start_pos: int, chunk_size: int) -> tuple[dict, int]:
    """
    Обрабатывает часть файла, возвращает индекс и позицию следующего чанка
    """
    questions_index = {}
    processed = 0
    next_pos = start_pos
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            f.seek(start_pos)
            
            while processed < chunk_size:
                pos = f.tell()
                line = f.readline()
                if not line:
                    next_pos = -1  # признак конца файла
                    break
                    
                try:
                    data = json.loads(line.strip())
                    question = data.get('question_text', '').strip()
                    
                    if question:
                        normalized_question = normalize_question(question)
                        questions_index[normalized_question] = {
                            'document_url': data.get('document_url', ''),
                            'example_id': data.get('example_id', ''),
                            'byte_offset': pos
                        }
                    
                    processed += 1
                    next_pos = f.tell()
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logging.error(f"Error processing line: {str(e)}")
                    continue
                
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        return {}, -1
    
    return questions_index, next_pos

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

def process_nq_open_chunk(input_file: str, output_file: str, unmatched_file: str,
                         questions_index: dict, simplified_nq_path: str,
                         start_line: int, chunk_size: int) -> tuple[int, int, int]:
    """Обрабатывает часть NQ-open датасета"""
    processed = matches = 0
    next_line = start_line
    
    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'a', encoding='utf-8') as fout, \
             open(unmatched_file, 'a', encoding='utf-8') as funmatched:
            
            # Пропускаем обработанные строки
            for _ in range(start_line):
                next(fin)
            
            pbar = tqdm(total=chunk_size, desc=f"Processing chunk from line {start_line}", 
                       unit=" questions")
            
            for line in fin:
                if processed >= chunk_size:
                    break
                    
                try:
                    data = json.loads(line.strip())
                    question = data.get('question', '').strip()
                    answer = data.get('answer', [])
                    normalized_question = normalize_question(question)
                    
                    if normalized_question in questions_index:
                        index_data = questions_index[normalized_question]
                        simplified_data = get_full_data(simplified_nq_path, index_data['byte_offset'])
                        
                        if simplified_data:
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
                    next_line += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'matches': f"{matches}/{processed}",
                        'rate': f"{matches/processed*100:.1f}%"
                    })
                
                except Exception as e:
                    logging.error(f"Error processing question: {str(e)}")
                    next_line += 1
                    continue
            
            pbar.close()
    
    except Exception as e:
        logging.error(f"Error processing file {input_file}: {str(e)}")
    
    return processed, matches, next_line

def main():
    parser = argparse.ArgumentParser(description='Process NQ-open dataset')
    parser.add_argument('--dataset', choices=['train', 'dev', 'both'], default='both',
                      help='Which dataset to process')
    parser.add_argument('--chunk-size', type=int, default=10000,
                      help='Process data in chunks of this size')
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
    
    # Очищаем выходные файлы
    for f in [train_output, dev_output, train_unmatched, dev_unmatched]:
        if os.path.exists(f):
            os.remove(f)
    
    # Обрабатываем данные частями
    chunk_start = 0
    total_processed = total_matches = 0
    
    while True:
        # Создаем индекс для текущего чанка
        logging.info(f"Processing chunk starting at position {chunk_start}")
        questions_index, next_chunk = process_chunk(simplified_nq, chunk_start, args.chunk_size)
        
        if not questions_index or next_chunk == -1:
            break
            
        logging.info(f"Created index for {len(questions_index)} questions in current chunk")
        
        # Обрабатываем датасеты с текущим индексом
        if args.dataset in ['train', 'both']:
            processed, matches, _ = process_nq_open_chunk(
                nq_open_train, train_output, train_unmatched,
                questions_index, simplified_nq, 0, args.chunk_size
            )
            total_processed += processed
            total_matches += matches
            logging.info(f"Chunk processed {processed} questions, found {matches} matches")
        
        if args.dataset in ['dev', 'both']:
            processed, matches, _ = process_nq_open_chunk(
                nq_open_dev, dev_output, dev_unmatched,
                questions_index, simplified_nq, 0, args.chunk_size
            )
            total_processed += processed
            total_matches += matches
            logging.info(f"Chunk processed {processed} questions, found {matches} matches")
        
        # Очищаем память
        questions_index.clear()
        gc.collect()
        
        # Переходим к следующему чанку
        chunk_start = next_chunk
        if next_chunk == -1:
            break
    
    end_time = datetime.now()
    logging.info(f"Processing completed in {end_time - start_time}")
    logging.info(f"Total processed: {total_processed}, total matches: {total_matches}")

if __name__ == "__main__":
    main()
