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

def process_simplified_nq(filepath: str) -> Dict[str, Tuple[str, str]]:
    """
    Обрабатывает упрощенную версию NQ датасета
    Возвращает словарь: вопрос -> (URL, исходный вопрос)
    """
    questions_dict = {}
    total_processed = 0
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    question = data.get('question_text', '').strip()
                    url = data.get('document_url', '')
                    
                    if question and url:
                        normalized_question = normalize_text(question)
                        questions_dict[normalized_question] = (url, question)
                        total_processed += 1
                        
                        if line_num % 10000 == 0:
                            logging.info(f"Processed {line_num} lines from simplified NQ dataset")
                
                except json.JSONDecodeError:
                    logging.error(f"Error parsing JSON at line {line_num}")
                    continue
                except Exception as e:
                    logging.error(f"Error processing line {line_num}: {str(e)}")
                    continue
    
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {str(e)}")
    
    logging.info(f"Finished processing simplified NQ dataset. Total questions: {total_processed}")
    return questions_dict

def process_efficient_qa(
    input_file: str,
    output_file: str,
    unmatched_file: str,
    nq_questions: Dict[str, Tuple[str, str]],
    similarity_threshold: float = 0.9
) -> Tuple[int, int, List[Tuple[str, str, str, str, float]]]:
    """
    Обрабатывает файл из efficient_qa датасета
    Возвращает: (обработано, найдено совпадений, примеры)
    """
    processed = 0
    matches_found = 0
    match_examples = []  # [(eq_question, eq_answer, nq_question, url, similarity)]
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout, \
         open(unmatched_file, 'w', encoding='utf-8') as funmatched:
        
        for line in fin:
            try:
                data = json.loads(line.strip())
                question = data.get('question', '').strip()
                answer = data.get('answer', [''])[0]
                
                # Нормализуем вопрос
                normalized_question = normalize_text(question)
                
                # Ищем лучшее совпадение
                best_match = None
                best_similarity = 0
                best_url = ''
                best_nq_question = ''
                
                # Сначала проверяем точное совпадение
                if normalized_question in nq_questions:
                    best_url, best_nq_question = nq_questions[normalized_question]
                    best_similarity = 1.0
                else:
                    # Если точного совпадения нет, ищем похожие
                    q1_keywords = get_keywords(normalized_question)
                    for nq_q, (url, orig_q) in nq_questions.items():
                        similarity = calculate_similarity(q1_keywords, get_keywords(nq_q))
                        if similarity >= similarity_threshold and similarity > best_similarity:
                            best_similarity = similarity
                            best_url = url
                            best_nq_question = orig_q
                
                if best_similarity >= similarity_threshold:
                    # Добавляем информацию из NQ
                    data['document_url'] = best_url
                    data['nq_similarity'] = best_similarity
                    data['nq_question'] = best_nq_question
                    matches_found += 1
                    
                    # Сохраняем пример для отчета
                    if len(match_examples) < 5:
                        match_examples.append((
                            question, answer, best_nq_question, best_url, best_similarity
                        ))
                else:
                    # Сохраняем ненайденный вопрос
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
                logging.error(f"Error parsing JSON in efficient_qa dataset")
                continue
            except Exception as e:
                logging.error(f"Error processing question: {str(e)}")
                continue
    
    return processed, matches_found, match_examples

def main():
    start_time = datetime.now()
    logging.info("Starting dataset processing")
    
    # Пути к файлам
    simplified_nq = 'simplified_qa/v1.0-simplified_simplified-nq-train.jsonl.gz'
    efficient_qa_dev = 'efficient_qa/NQ-open.efficientqa.dev.1.1.jsonl'
    efficient_qa_test = 'efficient_qa/NQ-open.efficientqa.test.1.1.jsonl'
    
    # Выходные файлы
    dev_output = 'efficient_qa/NQ-open.efficientqa.dev.with_refs.1.1.jsonl'
    test_output = 'efficient_qa/NQ-open.efficientqa.test.with_refs.1.1.jsonl'
    dev_unmatched = 'efficient_qa/unmatched_questions_dev.txt'
    test_unmatched = 'efficient_qa/unmatched_questions_test.txt'
    
    # Читаем упрощенный NQ датасет
    logging.info(f"Processing simplified NQ dataset: {simplified_nq}")
    nq_questions = process_simplified_nq(simplified_nq)
    logging.info(f"Loaded {len(nq_questions)} questions from simplified NQ dataset")
    
    # Обрабатываем efficient_qa dev датасет
    logging.info("Processing efficient_qa dev dataset...")
    dev_processed, dev_matches, dev_examples = process_efficient_qa(
        efficient_qa_dev, dev_output, dev_unmatched, nq_questions
    )
    
    # Обрабатываем efficient_qa test датасет
    logging.info("Processing efficient_qa test dataset...")
    test_processed, test_matches, test_examples = process_efficient_qa(
        efficient_qa_test, test_output, test_unmatched, nq_questions
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
        f.write(f"Total questions loaded: {len(nq_questions)}\n\n")
        
        f.write("Efficient QA Dev Dataset\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total questions: {dev_processed}\n")
        f.write(f"Matches found: {dev_matches}\n")
        f.write(f"Match rate: {dev_matches/dev_processed*100:.1f}%\n\n")
        
        f.write("Example matches from dev:\n")
        for q1, a1, q2, url, sim in dev_examples:
            f.write(f"\nEfficient QA Question: {q1}\n")
            f.write(f"Answer: {a1}\n")
            f.write(f"NQ Question: {q2}\n")
            f.write(f"URL: {url}\n")
            f.write(f"Similarity: {sim:.2f}\n")
            f.write("-" * 40 + "\n")
        
        f.write("\nEfficient QA Test Dataset\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total questions: {test_processed}\n")
        f.write(f"Matches found: {test_matches}\n")
        f.write(f"Match rate: {test_matches/test_processed*100:.1f}%\n\n")
        
        f.write("Example matches from test:\n")
        for q1, a1, q2, url, sim in test_examples:
            f.write(f"\nEfficient QA Question: {q1}\n")
            f.write(f"Answer: {a1}\n")
            f.write(f"NQ Question: {q2}\n")
            f.write(f"URL: {url}\n")
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
