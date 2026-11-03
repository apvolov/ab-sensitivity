# Данные WoWAH

В репозитории нет исходного файла `wowah_data.csv` (около 640 МБ). Набор World of Warcraft Avatar History распространяется отдельно и охватывает записи за 2008 год по персонажам игрового мира.

Ожидаются столбцы:

```text
char, level, race, charclass, zone, guild, timestamp
```

Скачайте набор из источника, указанного в исходном описании проекта: <https://www.kaggle.com/datasets/mylesoneill/warcraft-avatar-history>. Поместите файл в `data/wowah_data.csv` или задайте переменную окружения `WOWAH_CSV`.

При первом запуске код создает локальный кэш `data/daily.parquet`. CSV и Parquet исключены из Git правилами `.gitignore`.

Перед расчетами проверьте столбцы и формат дат функцией `peek()` из `ab_sensitivity.wowah`. В принятой конфигурации интервал опроса оценивается в 10 минут; медиана фактического интервала в исходном анализе составила 593 секунды.
