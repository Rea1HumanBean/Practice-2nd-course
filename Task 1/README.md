# Поиск цифрового следа пользователя в ВК

## Описание проекта
Проект предназначен для анализа активности пользователя ВКонтакте (VK) с целью оценки его эмоционального состояния на основе постов и комментариев. Мы будем работать с VK API для сбора данных и использовать методы машинного обучения (ML) для анализа текстов.

## Задачи
1. Получить все посты и комментарии пользователя в открытых сообществах и открытых профилях.
2. Оценить эмоциональное состояние пользователя по собранным данным с помощью методов машинного обучения:
    - Определить, находится ли пользователь в хорошем настроении, расстроен и т.д.
    - Определить количество состояний самостоятельно.

### Установка 
1.Склонировать репозиторий
        git clone https://github.com/Rea1HumanBean/Practice-2nd-course.git
        cd Practice-2nd-course/Task%201
2.Создать и активировать виртуальное окружение
        python -m venv .venv
3.Создать исполняемый файл
    -pyinstaller --onefile main.py
