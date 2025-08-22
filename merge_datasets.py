import json
import gzip
import os
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import string
import re
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

def normalize_text(text: str) -> str:
    """
    Базовая нормализация текста
    """
    text = text.lower().strip()
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    text = ' '.join(text.split())
    return text

def get_keywords(text: str) -> Set[str]:
    """
    Извлекает ключевые слова из текста
    """
    text = normalize_text(text)
    stop_words = {'a', 'an', 'the', 'is', 'was', 'were', 'will', 'be', 'in', 'on', 'at', 'to', 'for', 'of'}
    words = [w for w in text.split() if w not in stop_words]
    return set(words)

def calculate_similarity(keywords1: Set[str], keywords2: Set[str]) -> float:
    """
    Вычисляет схожесть между двумя наборами ключевых слов
    """
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    
    return intersection / union

def process_nq_file(filepath: str, questions_by_keyword: Dict[str, List[Tuple[str, str, Set[str]]]]) -> int:
    """
    Обрабатывает один файл из NQ датасета
    """
    processed = 0
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    question = data.get('question_text', '').strip()
                    url = data.get('document_url', '')
                    
                    if question and url:
                        keywords = get_keywords(question)
                        
                        # Для каждого ключевого слова сохраняем связь с вопросом
                        for keyword in keywords:
                            questions_by_keyword[keyword].append((question, url, keywords))
                        processed += 1
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logging.error(f"Error processing line in {filepath}: {str(e)}")
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {str(e)}")
    
    return processed

def find_matches(
    target_question: str,
    questions_by_keyword: Dict[str, List[Tuple[str, str, Set[str]]]],
    threshold: float = 0.9  # Повышенный порог схожести
) -> List[Tuple[str, str, float]]:
    """
    Ищет похожие вопросы
    """
    target_keywords = get_keywords(target_question)
    candidates = {}  # question -> (url, similarity)
    
    # Собираем кандидатов по каждому ключевому слову
    for keyword in target_keywords:
        if keyword in questions_by_keyword:
            for q, url, keywords in questions_by_keyword[keyword]:
                similarity = calculate_similarity(target_keywords, keywords)
                if similarity >= threshold:
                    if q not in candidates or candidates[q][1] < similarity:
                        candidates[q] = (url, similarity)
    
    # Сортируем результаты по убыванию схожести
    results = [(q, url, sim) for q, (url, sim) in candidates.items()]
    return sorted(results, key=lambda x: x[2], reverse=True)

def process_efficient_qa(
    input_file: str,
    output_file: str,
    unmatched_file: str,
    questions_by_keyword: Dict[str, List[Tuple[str, str, Set[str]]]]
) -> Tuple[int, int, List[Tuple[str, str, float]]]:
    """
    Обрабатывает файл из efficient_qa датасета
    """
    processed = 0
    matches_found = 0
    match_examples = []  # Сохраняем примеры совпадений для логов
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout, \
         open(unmatched_file, 'w', encoding='utf-8') as funmatched:
        
        for line in fin:
            try:
                data = json.loads(line.strip())
                question = data.get('question', '').strip()
                answer = data.get('answer', [''])[0]  # Сохраняем ответ для логов
                
                # Ищем похожие вопросы
                similar = find_matches(question, questions_by_keyword)
                
                if similar:
                    # Берем URL из лучшего совпадения
                    best_match = similar[0]
                    data['document_url'] = best_match[1]
                    data['nq_similarity'] = best_match[2]
                    data['nq_question'] = best_match[0]
                    matches_found += 1
                    
                    # Сохраняем пример для логов
                    if len(match_examples) < 5:  # Сохраняем только первые 5 примеров
                        match_examples.append((
                            question,
                            answer,
                            best_match[0],  # NQ вопрос
                            best_match[2]   # схожесть
                        ))
                else:
                    funmatched.write(json.dumps({
                        'question': question,
                        'answer': answer
                    }, ensure_ascii=False) + '\n')
                
                # Записываем обновленные данные
                fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed += 1
                
                if processed % 100 == 0:
                    logging.info(f"Processed {processed} questions, found matches for {matches_found}")
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logging.error(f"Error processing line: {str(e)}")
    
    return processed, matches_found, match_examples

def main():
    start_time = datetime.now()
    logging.info("Starting dataset processing")
    
    # Индексируем NQ датасет
    logging.info("Building index from NQ dataset...")
    questions_by_keyword = defaultdict(list)
    total_nq_processed = 0
    
    # Обрабатываем train файлы
    train_dir = 'v1.0/train'
    train_files = sorted([f for f in os.listdir(train_dir) if f.endswith('.jsonl.gz')])
    for i, filename in enumerate(train_files):
        filepath = os.path.join(train_dir, filename)
        logging.info(f"Processing {filename} ({i+1}/{len(train_files)})")
        processed = process_nq_file(filepath, questions_by_keyword)
        total_nq_processed += processed
    
    # Обрабатываем dev файлы
    dev_dir = 'v1.0/dev'
    dev_files = sorted([f for f in os.listdir(dev_dir) if f.endswith('.jsonl.gz')])
    for i, filename in enumerate(dev_files):
        filepath = os.path.join(dev_dir, filename)
        logging.info(f"Processing {filename} ({i+1}/{len(dev_files)})")
        processed = process_nq_file(filepath, questions_by_keyword)
        total_nq_processed += processed
    
    logging.info(f"Index built. Processed {total_nq_processed} questions from NQ dataset")
    logging.info(f"Total unique keywords: {len(questions_by_keyword)}")
    
    # Обрабатываем efficient_qa dev датасет
    logging.info("Processing efficient_qa dev dataset...")
    dev_input = 'efficient_qa/NQ-open.efficientqa.dev.1.1.jsonl'
    dev_output = 'efficient_qa/NQ-open.efficientqa.dev.with_refs.1.1.jsonl'
    dev_unmatched = 'efficient_qa/unmatched_questions_dev.txt'
    
    dev_processed, dev_matches, dev_examples = process_efficient_qa(
        dev_input, dev_output, dev_unmatched, questions_by_keyword
    )
    
    # Обрабатываем efficient_qa test датасет
    logging.info("Processing efficient_qa test dataset...")
    test_input = 'efficient_qa/NQ-open.efficientqa.test.1.1.jsonl'
    test_output = 'efficient_qa/NQ-open.efficientqa.test.with_refs.1.1.jsonl'
    test_unmatched = 'efficient_qa/unmatched_questions_test.txt'
    
    test_processed, test_matches, test_examples = process_efficient_qa(
        test_input, test_output, test_unmatched, questions_by_keyword
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
        
        f.write("NQ Dataset Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total questions processed: {total_nq_processed}\n")
        f.write(f"Unique keywords extracted: {len(questions_by_keyword)}\n\n")
        
        f.write("Efficient QA Dev Dataset\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total questions: {dev_processed}\n")
        f.write(f"Matches found: {dev_matches}\n")
        f.write(f"Match rate: {dev_matches/dev_processed*100:.1f}%\n\n")
        
        f.write("Example matches from dev:\n")
        for q1, a1, q2, sim in dev_examples:
            f.write(f"\nEfficient QA Question: {q1}\n")
            f.write(f"Answer: {a1}\n")
            f.write(f"NQ Question: {q2}\n")
            f.write(f"Similarity: {sim:.2f}\n")
            f.write("-" * 40 + "\n")
        
        f.write("\nEfficient QA Test Dataset\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total questions: {test_processed}\n")
        f.write(f"Matches found: {test_matches}\n")
        f.write(f"Match rate: {test_matches/test_processed*100:.1f}%\n\n")
        
        f.write("Example matches from test:\n")
        for q1, a1, q2, sim in test_examples:
            f.write(f"\nEfficient QA Question: {q1}\n")
            f.write(f"Answer: {a1}\n")
            f.write(f"NQ Question: {q2}\n")
            f.write(f"Similarity: {sim:.2f}\n")
            f.write("-" * 40 + "\n")
    
    # Выводим итоговую статистику
    logging.info("\nProcessing completed!")
    logging.info(f"Processing time: {processing_time}")
    logging.info(f"Dev dataset: processed {dev_processed} questions, found matches for {dev_matches}")
    logging.info(f"Test dataset: processed {test_processed} questions, found matches for {test_matches}")
    logging.info("Detailed report saved to processing_report.txt")

if __name__ == "__main__":
    main()