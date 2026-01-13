import json
import gzip
import logging
from datetime import datetime
from typing import Dict, Any, List, Set
import os
import argparse

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

def load_simplified_nq_minimal(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Загружает минимально необходимые данные из Simplified NQ датасета
    Возвращает словарь: нормализованный вопрос -> {document_url, example_id, original_question}
    """
    questions_dict = {}
    total_processed = 0
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    # Используем правильное поле question_text
                    question = data.get('question_text', '').strip()
                    
                    if question:
                        normalized_question = normalize_question(question)
                        # Сохраняем только необходимые поля и оригинальный вопрос
                        questions_dict[normalized_question] = {
                            'document_url': data.get('document_url', ''),
                            'example_id': data.get('example_id', ''),
                            'original_question': question  # сохраняем оригинальный вопрос для отладки
                        }
                        total_processed += 1
                        
                        if line_num % 10000 == 0:
                            logging.info(f"Processed {line_num} lines from simplified NQ dataset")
                            # Выводим пример для отладки
                            if line_num == 10000:
                                logging.info(f"Example normalized question: '{normalized_question}'")
                
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

def get_full_data_from_simplified_nq(filepath: str, target_question: str) -> Dict[str, Any]:
    """
    Ищет полные данные для конкретного вопроса в Simplified NQ
    """
    normalized_target = normalize_question(target_question)
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    # Используем правильное поле question_text
                    question = data.get('question_text', '').strip()
                    
                    if question and normalize_question(question) == normalized_target:
                        return data
                
                except (json.JSONDecodeError, Exception):
                    continue
    
    except Exception:
        pass
    
    return {}

def process_nq_open_batch(
    input_file: str,
    output_file: str,
    unmatched_file: str,
    simplified_nq_minimal: Dict[str, dict],
    simplified_nq_path: str,
    batch_size: int = 1000
) -> tuple[int, int]:
    """
    Обрабатывает файл из NQ-open датасета батчами
    Возвращает: (обработано, найдено совпадений)
    """
    processed = 0
    matches_found = 0
    current_batch = []
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout, \
         open(unmatched_file, 'w', encoding='utf-8') as funmatched:
        
        for line in fin:
            try:
                data = json.loads(line.strip())
                current_batch.append(data)
                
                # Обрабатываем батч
                if len(current_batch) >= batch_size:
                    batch_processed, batch_matches = process_batch(
                        current_batch,
                        fout,
                        funmatched,
                        simplified_nq_minimal,
                        simplified_nq_path
                    )
                    processed += batch_processed
                    matches_found += batch_matches
                    current_batch = []
                    logging.info(f"Processed {processed} questions, found matches for {matches_found}")
            
            except (json.JSONDecodeError, Exception) as e:
                logging.error(f"Error processing question in NQ-open: {str(e)}")
                continue
        
        # Обрабатываем оставшийся батч
        if current_batch:
            batch_processed, batch_matches = process_batch(
                current_batch,
                fout,
                funmatched,
                simplified_nq_minimal,
                simplified_nq_path
            )
            processed += batch_processed
            matches_found += batch_matches
    
    return processed, matches_found

def process_batch(
    batch: List[dict],
    fout,
    funmatched,
    simplified_nq_minimal: Dict[str, dict],
    simplified_nq_path: str
) -> tuple[int, int]:
    """Обрабатывает один батч данных"""
    processed = 0
    matches_found = 0
    
    for data in batch:
        try:
            # В NQ-open используется поле 'question'
            question = data.get('question', '').strip()
            answer = data.get('answer', [])
            normalized_question = normalize_question(question)
            
            # Для отладки
            if processed < 5:
                logging.debug(f"Processing question: '{question}'")
                logging.debug(f"Normalized question: '{normalized_question}'")
            
            # Проверяем наличие вопроса в минимальном словаре
            if normalized_question in simplified_nq_minimal:
                # Получаем полные данные только для найденных совпадений
                simplified_data = get_full_data_from_simplified_nq(simplified_nq_path, question)
                
                if simplified_data:
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
                else:
                    # Если почему-то не удалось получить полные данные
                    funmatched.write(json.dumps({
                        'question': question,
                        'answer': answer,
                        'error': 'Failed to get full data'
                    }, ensure_ascii=False) + '\n')
            else:
                # Сохраняем ненайденный вопрос
                funmatched.write(json.dumps({
                    'question': question,
                    'answer': answer,
                    'error': 'No match found'
                }, ensure_ascii=False) + '\n')
            
            processed += 1
            
            # Для отладки первых нескольких вопросов
            if processed <= 5:
                if normalized_question in simplified_nq_minimal:
                    logging.info(f"Match found for question: '{question}'")
                    logging.info(f"Simplified NQ question: '{simplified_nq_minimal[normalized_question]['original_question']}'")
                else:
                    logging.info(f"No match found for question: '{question}'")
        
        except Exception as e:
            logging.error(f"Error processing question: {str(e)}")
            continue
    
    return processed, matches_found

def main():
    parser = argparse.ArgumentParser(description='Process NQ-open dataset')
    parser.add_argument('--dataset', choices=['train', 'dev', 'both'], default='both',
                      help='Which dataset to process')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Batch size for processing')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
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
    
    # Загружаем минимальный словарь из simplified NQ
    logging.info(f"Loading minimal data from simplified NQ dataset: {simplified_nq}")
    simplified_nq_minimal = load_simplified_nq_minimal(simplified_nq)
    logging.info(f"Loaded {len(simplified_nq_minimal)} questions from simplified NQ dataset")
    
    # Выводим несколько примеров вопросов для отладки
    sample_questions = list(simplified_nq_minimal.items())[:5]
    logging.info("Sample questions from simplified NQ:")
    for norm_q, data in sample_questions:
        logging.info(f"Normalized: '{norm_q}' -> Original: '{data['original_question']}'")
    
    train_processed = train_matches = dev_processed = dev_matches = 0
    
    # Обрабатываем выбранные датасеты
    if args.dataset in ['train', 'both']:
        logging.info("Processing NQ-open train dataset...")
        train_processed, train_matches = process_nq_open_batch(
            nq_open_train, train_output, train_unmatched,
            simplified_nq_minimal, simplified_nq, args.batch_size
        )
    
    if args.dataset in ['dev', 'both']:
        logging.info("Processing NQ-open dev dataset...")
        dev_processed, dev_matches = process_nq_open_batch(
            nq_open_dev, dev_output, dev_unmatched,
            simplified_nq_minimal, simplified_nq, args.batch_size
        )
    
    # Записываем отчет
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
        f.write(f"Total questions loaded: {len(simplified_nq_minimal)}\n\n")
        
        if args.dataset in ['train', 'both']:
            f.write("NQ-open Train Dataset\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total questions: {train_processed}\n")
            f.write(f"Matches found: {train_matches}\n")
            f.write(f"Unmatched questions: {train_processed - train_matches}\n")
            f.write(f"Match rate: {train_matches/train_processed*100:.1f}%\n\n")
        
        if args.dataset in ['dev', 'both']:
            f.write("\nNQ-open Dev Dataset\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total questions: {dev_processed}\n")
            f.write(f"Matches found: {dev_matches}\n")
            f.write(f"Unmatched questions: {dev_processed - dev_matches}\n")
            f.write(f"Match rate: {dev_matches/dev_processed*100:.1f}%\n\n")
    
    # Выводим итоговую статистику
    logging.info("\nProcessing completed!")
    logging.info(f"Processing time: {processing_time}")
    if args.dataset in ['train', 'both']:
        logging.info(f"Train dataset: processed {train_processed} questions, found matches for {train_matches}")
    if args.dataset in ['dev', 'both']:
        logging.info(f"Dev dataset: processed {dev_processed} questions, found matches for {dev_matches}")
    logging.info("Detailed report saved to processing_report.txt")

if __name__ == "__main__":
    main()
