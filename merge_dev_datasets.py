import json
import logging
from datetime import datetime
import os
from tqdm import tqdm

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dev_processing_report.txt'),
        logging.StreamHandler()
    ]
)

def normalize_question(text: str) -> str:
    """Нормализация текста вопроса"""
    return text.lower().strip()

def load_simplified_nq(filepath: str) -> dict:
    """Загружает данные из упрощенного NQ dev датасета"""
    questions_dict = {}
    processed = 0
    
    logging.info(f"Starting to read {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Создаем progress bar без предварительного подсчета строк
            pbar = tqdm(desc="Loading simplified NQ dev", unit=" questions")
            
            for line in f:
                try:
                    data = json.loads(line.strip())
                    question = data.get('question', '').strip()
                    
                    if question:
                        normalized_question = normalize_question(question)
                        questions_dict[normalized_question] = data
                    
                    processed += 1
                    if processed % 1000 == 0:
                        pbar.update(1000)
                        pbar.set_postfix({'loaded': len(questions_dict)})
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logging.error(f"Error processing line: {str(e)}")
                    continue
            
            pbar.close()
            
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        return questions_dict
    
    logging.info(f"Finished loading. Processed {processed} lines, loaded {len(questions_dict)} questions")
    return questions_dict

def process_nq_open(input_file: str, output_file: str, unmatched_file: str, simplified_nq_data: dict):
    """Обрабатывает файл из NQ-open датасета"""
    processed = matches = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout, \
             open(unmatched_file, 'w', encoding='utf-8') as funmatched:
            
            # Создаем progress bar
            pbar = tqdm(desc=f"Processing {os.path.basename(input_file)}", unit=" questions")
            
            for line in fin:
                try:
                    data = json.loads(line.strip())
                    question = data.get('question', '').strip()
                    answer = data.get('answer', [])
                    normalized_question = normalize_question(question)
                    
                    if normalized_question in simplified_nq_data:
                        # Получаем данные из simplified NQ
                        simplified_data = simplified_nq_data[normalized_question]
                        
                        # Создаем новую запись
                        merged_data = {
                            'question': question,
                            'answer': answer,
                            'document_text': simplified_data.get('document_text', ''),
                            'document_url': simplified_data.get('document_url', ''),
                            'annotations': simplified_data.get('annotations', []),
                            'long_answer_candidates': simplified_data.get('long_answer_candidates', []),
                            'example_id': simplified_data.get('example_id', '')
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
    start_time = datetime.now()
    logging.info("Starting dev dataset processing")
    
    # Пути к файлам
    simplified_nq_dev = 'nq_open/NQ-open.dev.jsonl'  # наш упрощенный dev датасет
    nq_open_dev = 'efficient_qa/NQ-open.efficientqa.dev.1.1.jsonl'  # NQ-open dev датасет
    
    # Выходные файлы
    dev_output = 'nq_open/NQ-open.dev.merged.jsonl'
    dev_unmatched = 'nq_open/unmatched_questions_dev.txt'
    
    # Загружаем данные из упрощенного NQ dev
    logging.info("Loading simplified NQ dev dataset...")
    simplified_nq_data = load_simplified_nq(simplified_nq_dev)
    if not simplified_nq_data:
        logging.error("Failed to load simplified NQ dev dataset")
        return
    
    logging.info(f"Loaded {len(simplified_nq_data)} questions from simplified NQ dev dataset")
    
    # Обрабатываем dev датасет
    logging.info("Processing dev dataset...")
    dev_processed, dev_matches = process_nq_open(
        nq_open_dev, dev_output, dev_unmatched, simplified_nq_data
    )
    logging.info(f"Dev dataset: processed {dev_processed}, matched {dev_matches}")
    
    end_time = datetime.now()
    logging.info(f"Processing completed in {end_time - start_time}")

if __name__ == "__main__":
    main()
