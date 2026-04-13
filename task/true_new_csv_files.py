import os
import pandas as pd

# Папка с исходными CSV файлами
input_folder = r'C:\Education\4 course 1 semester\Practice\Synchronization task\Data\csv files'  # замените на путь к вашей папке
# Папка, куда будут сохраняться изменённые CSV
output_folder = r'C:\Education\4 course 1 semester\Practice\Synchronization task\Data\new csv files'  # замените на путь к вашей папке

# Если папки назначения не существует, создаём её
os.makedirs(output_folder, exist_ok=True)

# Проходим по всем CSV файлам в исходной папке
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Считываем CSV
        df = pd.read_csv(input_path)

        # Проверяем, есть ли столбец 'time'
        if 'time' in df.columns:
            # Добавляем 3 часа (3 * 3600 секунд)
            df['time'] = df['time'] + 3 * 3600

        # Сохраняем изменённый CSV
        df.to_csv(output_path, index=False)

print("Готово! Все файлы обработаны.")
