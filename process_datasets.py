import json
import gzip
import os
import sqlite3
from tqdm import tqdm
from typing import Set, Tuple, Dict, Any
import logging
import re

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

def normalize_question(question: str) -> str:
    """
    Нормализация вопроса для лучшего сопоставления
    """
    # Приводим к нижнему регистру
    question = question.lower().strip()
    
    # Удаляем знаки препинания
    question = re.sub(r'[.,?!]', '', question)
    
    # Удаляем множественные пробелы
    question = ' '.join(question.split())
    
    # Удаляем вопросительные слова в начале
    question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 'whose', 'whom']
    words = question.split()
    if words and words[0] in question_words:
        question = ' '.join(words[1:])
        
    # Если после what is/was/were, убираем и это
    question = re.sub(r'^(is|was|were)\s+', '', question)
    
    return question

def init_database(db_path: str):
    """
    Инициализация SQLite базы данных для хранения данных из NQ
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Создаем таблицу для хранения всех данных в формате JSON
    c.execute('''CREATE TABLE IF NOT EXISTS question_data
                 (question TEXT PRIMARY KEY, original_question TEXT, data_json TEXT)''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_question ON question_data(question)')
    conn.commit()
    return conn

def process_nq_file(filepath: str, conn: sqlite3.Connection, batch_size: int = 1000):
    """
    Обработка одного NQ файла и сохранение всех данных в SQLite
    """
    batch = []
    cursor = conn.cursor()
    processed = 0
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # В NQ датасете вопрос находится в поле question_text
                    question = data.get('question_text', '').strip()
                    
                    if question:
                        # Сохраняем оригинальный вопрос и нормализованную версию
                        normalized_question = normalize_question(question)
                        
                        # Сохраняем все поля кроме тех, что есть в efficient_qa
                        stored_data = {
                            'document_url': data.get('document_url', ''),
                            'document_title': data.get('document_title', ''),
                            'annotations': data.get('annotations', []),
                            'long_answer_candidates': data.get('long_answer_candidates', [])
                        }
                        
                        batch.append((
                            normalized_question,
                            question,  # оригинальный вопрос
                            json.dumps(stored_data, ensure_ascii=False)
                        ))
                        processed += 1
                        
                        if len(batch) >= batch_size:
                            cursor.executemany(
                                'INSERT OR REPLACE INTO question_data (question, original_question, data_json) VALUES (?, ?, ?)',
                                batch
                            )
                            conn.commit()
                            batch = []
                            
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logging.error(f"Error processing line in {filepath}: {str(e)}")
                    continue
            
            # Сохраняем оставшиеся записи
            if batch:
                cursor.executemany(
                    'INSERT OR REPLACE INTO question_data (question, original_question, data_json) VALUES (?, ?, ?)',
                    batch
                )
                conn.commit()
                
    except Exception as e:
        logging.error(f"Error processing file {filepath}: {str(e)}")
    
    return processed

def read_nq_dataset(nq_dir: str, db_path: str):
    """
    Чтение всего NQ датасета и сохранение в SQLite
    """
    conn = init_database(db_path)
    total_processed = 0
    
    # Обработка всех файлов (train и dev) как единый источник данных
    directories = ['train', 'dev']
    
    for dir_name in directories:
        dir_path = os.path.join(nq_dir, dir_name)
        files = [f for f in os.listdir(dir_path) if f.endswith('.jsonl.gz')]
        
        for filename in tqdm(files, desc=f"Processing {dir_name} files"):
            filepath = os.path.join(dir_path, filename)
            processed = process_nq_file(filepath, conn)
            total_processed += processed
            logging.info(f"Processed file: {filename} (found {processed} questions)")
    
    logging.info(f"Total questions processed: {total_processed}")
    
    # Выводим примеры нормализации
    cursor = conn.cursor()
    cursor.execute('SELECT question, original_question FROM question_data LIMIT 5')
    examples = cursor.fetchall()
    logging.info("Examples of question normalization:")
    for norm, orig in examples:
        logging.info(f"Original : {orig}")
        logging.info(f"Normalized: {norm}")
        logging.info("---")
    
    return conn

def process_efficient_qa_batch(
    input_file: str,
    output_file: str,
    unmatched_file: str,
    db_conn: sqlite3.Connection,
    batch_size: int = 1000
):
    """
    Обработка efficient_qa датасета батчами
    """
    cursor = db_conn.cursor()
    questions_without_refs = set()
    questions_with_refs = set()
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        batch = []
        for line in tqdm(fin, desc=f"Processing {os.path.basename(input_file)}"):
            try:
                data = json.loads(line.strip())
                question = data.get('question', '').strip()
                normalized_question = normalize_question(question)
                
                # Проверяем наличие данных в базе
                cursor.execute('SELECT original_question, data_json FROM question_data WHERE question = ?', 
                             (normalized_question,))
                result = cursor.fetchone()
                
                if result:
                    # Добавляем все дополнительные поля из NQ датасета
                    original_question, additional_data_json = result
                    additional_data = json.loads(additional_data_json)
                    data.update(additional_data)
                    data['nq_original_question'] = original_question  # сохраняем оригинальный вопрос для анализа
                    questions_with_refs.add(question)
                else:
                    questions_without_refs.add(question)
                
                batch.append(data)
                
                if len(batch) >= batch_size:
                    # Записываем батч
                    for item in batch:
                        fout.write(json.dumps(item, ensure_ascii=False) + '\n')
                    batch = []
                    
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON in {input_file}")
                continue
            except Exception as e:
                logging.error(f"Error processing line in {input_file}: {str(e)}")
                continue
        
        # Записываем оставшийся батч
        if batch:
            for item in batch:
                fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Сохраняем список вопросов без совпадений
    with open(unmatched_file, 'w', encoding='utf-8') as f:
        for question in sorted(questions_without_refs):
            f.write(question + '\n')
    
    logging.info(f"Found matches for {len(questions_with_refs)} questions")
    logging.info(f"No matches for {len(questions_without_refs)} questions")
    
    # Выводим примеры успешных совпадений
    if questions_with_refs:
        logging.info("Examples of successful matches:")
        cursor.execute('''
            SELECT q.question, q.original_question 
            FROM question_data q 
            WHERE q.question IN (
                SELECT normalize_question(?) 
                FROM (SELECT ? AS q UNION ALL SELECT ? UNION ALL SELECT ? UNION ALL SELECT ? UNION ALL SELECT ?) t
            )
        ''', list(questions_with_refs)[:5] * 6)
        matches = cursor.fetchall()
        for norm, orig in matches:
            logging.info(f"Efficient QA: {norm}")
            logging.info(f"NQ Original: {orig}")
            logging.info("---")
    
    # Выводим примеры вопросов без совпадений
    if questions_without_refs:
        logging.info("Examples of questions without matches:")
        for q in sorted(list(questions_without_refs))[:5]:
            logging.info(f"  - {q}")
            logging.info(f"  Normalized: {normalize_question(q)}")
            logging.info("---")
    
    return len(questions_without_refs)

def main():
    # Конфигурация путей
    nq_dir = 'v1.0'
    efficient_qa_dev = 'efficient_qa/NQ-open.efficientqa.dev.1.1.jsonl'
    efficient_qa_test = 'efficient_qa/NQ-open.efficientqa.test.1.1.jsonl'
    
    # Пути для выходных файлов
    output_dev = 'efficient_qa/NQ-open.efficientqa.dev.with_refs.1.1.jsonl'
    output_test = 'efficient_qa/NQ-open.efficientqa.test.with_refs.1.1.jsonl'
    unmatched_questions_dev = 'efficient_qa/unmatched_questions_dev.txt'
    unmatched_questions_test = 'efficient_qa/unmatched_questions_test.txt'
    
    # Путь к базе данных SQLite
    db_path = 'question_refs.db'
    
    try:
        # Удаляем старую базу данных, если она существует
        if os.path.exists(db_path):
            os.remove(db_path)
            logging.info("Removed old database")
        
        logging.info("Creating new database and processing NQ dataset...")
        db_conn = read_nq_dataset(nq_dir, db_path)
        
        # Обработка dev датасета
        logging.info("Processing efficient_qa dev set...")
        dev_unmatched = process_efficient_qa_batch(
            efficient_qa_dev, output_dev, unmatched_questions_dev, db_conn
        )
        
        # Обработка test датасета
        logging.info("Processing efficient_qa test set...")
        test_unmatched = process_efficient_qa_batch(
            efficient_qa_test, output_test, unmatched_questions_test, db_conn
        )
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
    finally:
        if 'db_conn' in locals():
            db_conn.close()

if __name__ == "__main__":
    main()