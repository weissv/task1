#!/bin/bash
# Упаковка решения в плоский zip-архив для contest.yandex.ru

set -e

if [ ! -d "weights" ] || [ -z "$(ls -A weights 2>/dev/null)" ]; then
    echo "ОШИБКА: папка weights/ пуста или не существует!"
    exit 1
fi

rm -f submission.zip

zip -r submission.zip \
    Dockerfile \
    solution.py \
    weights/

echo "Готово! submission.zip создан."
echo "Размер: $(du -sh submission.zip | cut -f1)"
