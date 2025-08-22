import json
import gzip
import os
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import string

def get_stop_words() -> Set[str]:
    """
    Возвращает список стоп-слов
    """
    return {
        'a', 'an', 'the', 'is', 'was', 'were', 'will', 'be', 'to', 'of', 'and',
        'in', 'on', 'at', 'by', 'for', 'with', 'about', 'from', 'did', 'does',
        'do', 'has', 'have', 'had', 'what', 'when', 'where', 'who', 'why', 'how',
        'which', 'whose', 'whom', 'that'
    }

def normalize_text(text: str) -> str:
    """
    Базовая нормализация текста
    """
    # Приводим к нижнему регистру
    text = text.lower().strip()
    # Заменяем пунктуацию на пробелы
    text = re.sub(f'[{string.punctuation}]', ' ', text)
    # Убираем множественные пробелы
    text = ' '.join(text.split())
    return text

def get_keywords(text: str, remove_stop_words: bool = True) -> Set[str]:
    """
    Извлекает ключевые слова из текста
    """
    text = normalize_text(text)
    words = text.split()
    if remove_stop_words:
        stop_words = get_stop_words()
        words = [w for w in words if w not in stop_words]
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

def process_nq_files(file_paths: List[str], num_files: int = 3) -> Dict[str, List[Tuple[str, str, Set[str]]]]:
    """
    Читает первые num_files файлов из NQ датасета
    """
    questions_by_keyword = defaultdict(list)
    total_processed = 0
    
    print(f"\nReading first {num_files} files from NQ dataset:")
    
    for filepath in file_paths[:num_files]:
        print(f"\nProcessing {filepath}:")
        
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    question = data.get('question_text', '').strip()
                    url = data.get('document_url', '')
                    
                    if question:
                        keywords = get_keywords(question)
                        
                        # Для каждого ключевого слова сохраняем связь с вопросом
                        for keyword in keywords:
                            questions_by_keyword[keyword].append((question, url, keywords))
                        
                        total_processed += 1
                        
                        if i <= 5:
                            print(f"Q{i}: {question}")
                            print(f"Keywords: {keywords}")
                            print(f"URL: {url}")
                            print("---")
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing line: {str(e)}")
    
    print(f"\nTotal questions processed: {total_processed}")
    print(f"Total unique keywords: {len(questions_by_keyword)}")
    return questions_by_keyword

def find_matches(
    target_question: str,
    questions_by_keyword: Dict[str, List[Tuple[str, str, Set[str]]]],
    threshold: float = 0.3
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
    questions_by_keyword: Dict[str, List[Tuple[str, str, Set[str]]]],
    num_questions: int = 50
):
    """
    Обрабатываем вопросы из efficient_qa
    """
    filepath = 'efficient_qa/NQ-open.efficientqa.dev.1.1.jsonl'
    matches_found = []
    questions_without_matches = []
    
    print("\nEfficient QA Dataset questions:")
    print("-" * 80)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_questions:
                break
                
            try:
                data = json.loads(line.strip())
                question = data.get('question', '').strip()
                answer = data.get('answer', [''])[0]
                
                print(f"\nQ{i+1}: {question}")
                print(f"Answer: {answer}")
                
                # Ищем похожие вопросы
                similar = find_matches(question, questions_by_keyword)
                
                if similar:
                    matches_found.append((question, similar[:3]))  # сохраняем топ-3 совпадения
                    print("Matches found:")
                    for j, (nq_q, url, sim) in enumerate(similar[:3], 1):
                        print(f"  {j}. Similarity: {sim:.2f}")
                        print(f"     NQ: {nq_q}")
                        print(f"     URL: {url}")
                else:
                    questions_without_matches.append(question)
                    print("No matches found")
                
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Error processing line: {str(e)}")
    
    return matches_found, questions_without_matches

def main():
    # Собираем пути к файлам NQ датасета
    nq_files = []
    
    # Добавляем файлы из train
    train_dir = 'v1.0/train'
    train_files = sorted([os.path.join(train_dir, f) for f in os.listdir(train_dir) 
                         if f.endswith('.jsonl.gz')])
    nq_files.extend(train_files[:2])  # Берем первые 2 файла из train
    
    # Добавляем файлы из dev
    dev_dir = 'v1.0/dev'
    dev_files = sorted([os.path.join(dev_dir, f) for f in os.listdir(dev_dir) 
                       if f.endswith('.jsonl.gz')])
    nq_files.extend(dev_files[:1])  # Берем первый файл из dev
    
    # Читаем NQ датасет
    questions_by_keyword = process_nq_files(nq_files)
    
    # Обрабатываем efficient_qa
    matches, no_matches = process_efficient_qa(questions_by_keyword)
    
    # Выводим статистику
    print("\nResults:")
    print(f"Questions with matches: {len(matches)}")
    print(f"Questions without matches: {len(no_matches)}")
    
    if matches:
        print("\nExample matches:")
        for i, (q, similar) in enumerate(matches[:5], 1):
            print(f"\n{i}. Efficient QA: {q}")
            for j, (nq_q, url, sim) in enumerate(similar, 1):
                print(f"   Match {j} (similarity: {sim:.2f}):")
                print(f"   NQ: {nq_q}")
                print(f"   URL: {url}")
    
    if no_matches:
        print("\nExample questions without matches:")
        for q in no_matches[:5]:
            print(f"- {q}")
            print(f"  Keywords: {get_keywords(q)}")

if __name__ == "__main__":
    main()