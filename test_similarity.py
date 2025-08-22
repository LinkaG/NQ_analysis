import json
from typing import Set
import string
import re

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

def test_similarity(q1: str, q2: str):
    """
    Тестирует схожесть двух вопросов
    """
    k1 = get_keywords(q1)
    k2 = get_keywords(q2)
    sim = calculate_similarity(k1, k2)
    
    print(f"Q1: {q1}")
    print(f"Keywords 1: {k1}")
    print(f"\nQ2: {q2}")
    print(f"Keywords 2: {k2}")
    print(f"\nCommon words: {k1 & k2}")
    print(f"All unique words: {k1 | k2}")
    print(f"Similarity: {sim:.2f}")
    print("-" * 80)

# Тестируем разные пары вопросов
test_cases = [
    # Очень похожие вопросы (должны совпадать)
    (
        "when did the harry potter movie come out",
        "when did harry potter the movie come out"
    ),
    
    # Похожие вопросы, но разный смысл
    (
        "who played apollo creed in the original rocky",
        "who played rocky in the original movie"
    ),
    
    # Похожие по структуре, но разные вопросы
    (
        "when was the last time liverpool won first 5 games",
        "when was the last time arsenal won the league"
    ),
    
    # Вопросы с общими словами, но разным смыслом
    (
        "who sings the song i don't care i love it",
        "who sings i got you under my skin"
    ),
]

print("Testing different similarity thresholds:\n")
for q1, q2 in test_cases:
    test_similarity(q1, q2)
