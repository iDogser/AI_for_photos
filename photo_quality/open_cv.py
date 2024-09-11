import cv2
import os
import numpy as np

# Путь к изображению и директории
input_dir = 'src_input'
output_dir = 'src_output'
input_image_path = os.path.join(input_dir, 'test1.jpg')
output_image_path = os.path.join(output_dir, 'processed_test1.jpg')

# Загрузка изображения
img = cv2.imread(input_image_path)

# Проверка, что изображение загружено правильно
if img is None:
    print(f"Ошибка: не удалось загрузить изображение по пути {input_image_path}")
else:
    # Преобразование в градации серого
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Применение фильтра размытия
    blurred_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Повышение резкости
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(blurred_img, -1, kernel)

    # Применение CLAHE для увеличения контрастности
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(200, 200))
    contrast_img = clahe.apply(sharpened_img)

    # Сохранение обработанного изображения в папку output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv2.imwrite(output_image_path, contrast_img)
    print(f"Обработанное изображение сохранено по пути {output_image_path}")
