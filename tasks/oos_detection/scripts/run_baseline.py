"""
Запуск бейзлайнов: TF-IDF + LogReg и Cosine Similarity.

Использование:
    # Бейзлайн A
    python scripts/run_baseline.py --model tfidf

    # Бейзлайн B — оба варианта модели одной командой
    python scripts/run_baseline.py --model cosine

    # Бейзлайн B — конкретная модель
    python scripts/run_baseline.py --model cosine --model_name bert-base-uncased
    python scripts/run_baseline.py --model cosine --model_name sentence-transformers/all-MiniLM-L6-v2

    # Все бейзлайны подряд
    python scripts/run_baseline.py --model all
"""

# TODO: реализовать
# Аргументы: --model (tfidf | cosine | all), --model_name, --split (easy|plus)
# Выход: results/metrics.json (дописать), results/error_analysis.csv
