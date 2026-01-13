import json
import gzip
import logging
from datetime import datetime
from typing import Dict, Any, List, Set
import os

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

def load_simplified_nq(filepath: str) -> Dict[str, dict]:
    """
    Загружает данные из Simplified NQ датасета
    Возвращает словарь: нормализованный вопрос -> полные данные
    """
    questions_dict = {}
    total_processed = 0
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    question = data.get('question_text', '').strip()
                    
                    if question:
                        normalized_question = normalize_question(question)
                        questions_dict[normalized_question] = data
                        total_processed += 1
                        
                        if line_num % 10000 == 0:
                            logging.info(f"Processed {line_num} lines from simplified NQ dataset")
                
                except json.JSONDecodeError:
                    logging.error(f"Error parsing JSON at line {line_num} in simplified NQ")
                    continue
                except Exception as e:
                    logging.error(f"Error processing line {line_num} in simplified NQ: {str(e)}")
                    continue
    
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {str(e)}")
    
    logging.info(f"Finished loading simplified NQ dataset. Total questions: {total_processed}")
    return questions_dict

def process_nq_open(
    input_file: str,
    output_file: str,
    unmatched_file: str,
    simplified_nq: Dict[str, dict]
) -> tuple[int, int, List[tuple[str, str]]]:
    """
    Обрабатывает файл из NQ-open датасета
    Возвращает: (обработано, найдено совпадений, примеры)
    """
    processed = 0
    matches_found = 0
    match_examples = []  # [(nq_open_question, simplified_nq_question)]
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout, \
         open(unmatched_file, 'w', encoding='utf-8') as funmatched:
        
        for line in fin:
            try:
                data = json.loads(line.strip())
                question = data.get('question', '').strip()
                answer = data.get('answer', [])
                
                # Нормализуем вопрос
                normalized_question = normalize_question(question)
                
                # Ищем соответствие в simplified NQ
                if normalized_question in simplified_nq:
                    # Получаем данные из simplified NQ
                    simplified_data = simplified_nq[normalized_question]
                    
                    # Создаем новую запись, объединяя данные
                    merged_data = {
                        'question': question,  # оригинальный вопрос из NQ-open
                        'answer': answer,      # оригинальный ответ из NQ-open
                        'document_text': simplified_data.get('document_text', ''),
                        'document_url': simplified_data.get('document_url', ''),
                        'annotations': simplified_data.get('annotations', []),
                        'long_answer_candidates': simplified_data.get('long_answer_candidates', []),
                        'example_id': simplified_data.get('example_id', '')
                    }
                    
                    # Записываем объединенные данные
                    fout.write(json.dumps(merged_data, ensure_ascii=False) + '\n')
                    matches_found += 1
                    
                    # Сохраняем пример для отчета
                    if len(match_examples) < 5:
                        match_examples.append((
                            question,
                            simplified_data.get('question_text', '')
                        ))
                else:
                    # Сохраняем ненайденный вопрос
                    funmatched.write(json.dumps({
                        'question': question,
                        'answer': answer
                    }, ensure_ascii=False) + '\n')
                
                processed += 1
                
                if processed % 1000 == 0:
                    logging.info(f"Processed {processed} questions from NQ-open, found matches for {matches_found}")
                
            except json.JSONDecodeError:
                logging.error(f"Error parsing JSON in NQ-open dataset")
                continue
            except Exception as e:
                logging.error(f"Error processing question in NQ-open: {str(e)}")
                continue
    
    return processed, matches_found, match_examples

def main():
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
    
    # Загружаем simplified NQ датасет
    logging.info(f"Loading simplified NQ dataset: {simplified_nq}")
    simplified_nq_data = load_simplified_nq(simplified_nq)
    logging.info(f"Loaded {len(simplified_nq_data)} questions from simplified NQ dataset")
    
    # Обрабатываем train датасет
    logging.info("Processing NQ-open train dataset...")
    train_processed, train_matches, train_examples = process_nq_open(
        nq_open_train, train_output, train_unmatched, simplified_nq_data
    )
    
    # Обрабатываем dev датасет
    logging.info("Processing NQ-open dev dataset...")
    dev_processed, dev_matches, dev_examples = process_nq_open(
        nq_open_dev, dev_output, dev_unmatched, simplified_nq_data
    )
    
    # Записываем подробный отчет
    end_time = datetime.now()
    processing_time = end_time - start_time
    
    with open('processing_report.txt', 'w', encoding='utf-8') as f:
        f.write("Processing Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Processing Time\n")
        f.write("-" * 80 + "\n")
        f.write(f"Started: {start_time}\n")
        f.write(f"Finished: {end_time}\n")
        f.write(f"Total time: {processing_time}\n\n")
        
        f.write("Simplified NQ Dataset Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total questions loaded: {len(simplified_nq_data)}\n\n")
        
        f.write("NQ-open Train Dataset\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total questions: {train_processed}\n")
        f.write(f"Matches found: {train_matches}\n")
        f.write(f"Unmatched questions: {train_processed - train_matches}\n")
        f.write(f"Match rate: {train_matches/train_processed*100:.1f}%\n\n")
        
        f.write("Example matches from train:\n")
        for q1, q2 in train_examples:
            f.write(f"\nNQ-open Question: {q1}\n")
            f.write(f"Simplified NQ Question: {q2}\n")
            f.write("-" * 40 + "\n")
        
        f.write("\nNQ-open Dev Dataset\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total questions: {dev_processed}\n")
        f.write(f"Matches found: {dev_matches}\n")
        f.write(f"Unmatched questions: {dev_processed - dev_matches}\n")
        f.write(f"Match rate: {dev_matches/dev_processed*100:.1f}%\n\n")
        
        f.write("Example matches from dev:\n")
        for q1, q2 in dev_examples:
            f.write(f"\nNQ-open Question: {q1}\n")
            f.write(f"Simplified NQ Question: {q2}\n")
            f.write("-" * 40 + "\n")
    
    # Выводим итоговую статистику
    logging.info("\nProcessing completed!")
    logging.info(f"Processing time: {processing_time}")
    logging.info(f"Train dataset: processed {train_processed} questions, found matches for {train_matches}")
    logging.info(f"Dev dataset: processed {dev_processed} questions, found matches for {dev_matches}")
    logging.info("Detailed report saved to processing_report.txt")

if __name__ == "__main__":
    main()