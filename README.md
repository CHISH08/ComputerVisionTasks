# 🧑‍🎓 Компьютерное зрение — Лабораторные работы

### Студент: Дегтярев Денис Андреевич, группа М8О-407Б-21

---

## 📁 Структура репозитория

```
ComputerVisionTasks/
│
├── .git/                         # Git-репозиторий
├── data/                         # Сырые и предобработанные датасеты
│   ├── Dior/
│   ├── PatternNet/
│   ├── PatternNet.zip
│   └── Лабораторные работы 6-8 (2024-2025).docx
│
├── models_implementation/        # Собственные реализации моделей
│   ├── __pycache__/
│   ├── __init__.py
│   ├── classification/           # Классификация
│   │   ├── __pycache__/
│   │   ├── resnet18.py
│   │   └── vitb16.py
│   ├── detection/                # Обнаружение объектов
│   └── segmentation/             # Сегментация
│       └── transunet.py
│
├── notebook/                     # Jupyter-ноутбуки по лабораторным
│   ├── lab6/                     # ЛР №6: Классификация (torchvision)
│   │   ├── torchvision_without_augmentation.ipynb
│   │   └── torchvision_with_augmentation.ipynb
│   │
│   ├── lab7/                     # ЛР №7: Семантическая сегментация (SMP)
│   │   ├── implementation_unet_vit_with_augmentation.ipynb
│   │   ├── smp_unet_cnn_with_augmentation.ipynb
│   │   └── smp_unet_vit_with_augmentation.ipynb
│   │
│   └── lab8/                     # ЛР №8: Обнаружение объектов (YOLOv11)
│       ├── my_implementation_yolo11.ipynb
│       └── ultralitycs_yolo11.ipynb
│
├── other_notebook/               # Дополнительный ноутбук для подготовки данных
│   └── make_classif_sample.ipynb
│
├── runs/                         # Результаты тренировок YOLO (веса, логи, графики)
│
├── .gitattributes
├── .gitignore
├── data.zip                      # Архив с исходными данными
├── environment.yml               # Окружение Conda
├── LICENSE                       # Лицензия MIT
└── README.md                     # Этот файл
```

---

## 🔎 Задание и порядок выполнения

В едином GitHub-репозитории представлены результаты трёх лабораторных работ. Модульная структура, Jupyter-ноутбуки снабжены комментариями в каждой ячейке, а в корне проекта — этот README.

### Лабораторная работа №6

**Проведение исследований с моделями классификации**

1. **Выбор начальных условий**
   a. Уникальный датасет для задачи классификации (+обоснование практической задачи)
   b. Выбор и обоснование метрик качества

2. **Создание бейзлайна и оценка качества**
   a. Обучение сверточных и «трансформерных» моделей из `torchvision`
   b. Оценка качества по выбранным метрикам

3. **Улучшение бейзлайна**
   a. Формулировка гипотез (аугментации, подбор моделей и гиперпараметров)
   b. Проверка гипотез
   c. Формирование улучшенного бейзлайна
   d. Обучение моделей с улучшенным бейзлайном
   e. Оценка качества
   f. Сравнение с исходным бейзлайном
   g. Выводы

4. **Имплементация алгоритмов**
   a. Своё кодирование моделей
   b. Обучение имплементированных моделей
   c. Оценка качества
   d. Сравнение с результатами пункта 2
   e. Выводы
   f. Добавление техник из пункта 3c
   g. Перетренировка и оценка
   h. Сравнение с пунктом 3
   i. Выводы

**Результаты ЛР6:**

* **ResNet‑18 (без аугментаций):** Accuracy≈75 %, F1≈0.74.
* **ViT‑B16 (без аугментаций):** Accuracy≈72 %, F1≈0.70.
* **ResNet‑18 (с аугментациями):** Accuracy≈85 %, F1≈0.84 (+10 %).
* **ViT‑B16 (с аугментациями):** Accuracy≈82 %, F1≈0.81 (+10 %).
* **Собственные реализации:** до Accuracy≈83 %, F1≈0.82 после аугментаций и тюнинга.

**Ключевые выводы:**

* Аугментации дают +10–12 % к качеству.
* CNN-архитектуры устойчивее на малых выборках, но трансформеры приближаются после Data Augmentation.

Ноутбуки ЛР6:

* `notebook/lab6/torchvision_without_augmentation.ipynb`
* `notebook/lab6/torchvision_with_augmentation.ipynb`

---

### Лабораторная работа №7

**Семантическая сегментация с использованием `segmentation_models.pytorch`**

Повторяются пункты 2–4 из ЛР6, но все модели реализованы через SMP.

**Метрики:** IoU, Dice Coefficient
**Результаты ЛР7:**

* **SMP Unet (CNN):** IoU≈0.65, Dice≈0.72
* **SMP Unet+ViT:** IoU≈0.70, Dice≈0.76 (+5–7 %)
* **Собственная реализация Unet+ViT:** IoU≈0.68, Dice≈0.74

**Ключевые выводы:**

* ViT-бекбон улучшает сегментацию на \~0.05 IoU.
* Аугментации улучшают границы объектов и повышают Dice.

Ноутбуки ЛР7:

* `notebook/lab7/implementation_unet_vit_with_augmentation.ipynb`
* `notebook/lab7/smp_unet_cnn_with_augmentation.ipynb`
* `notebook/lab7/smp_unet_vit_with_augmentation.ipynb`

---

### Лабораторная работа №8

**Обнаружение и распознавание объектов с помощью `ultralytics` (YOLOv11)**

Повторяются пункты 2–4 из ЛР6, но используются модели YOLOv11.

**Метрики:** mAP\@0.5, Precision, Recall
**Результаты ЛР8:**

* **YOLOv11 (стандартная тренировка):** mAP\@0.5≈0.60, Precision≈0.65, Recall≈0.62
* **YOLOv11 (тюнинг + аугментации):** mAP\@0.5≈0.67, Precision≈0.70, Recall≈0.68 (+7 %)

**Ключевые выводы:**

* Гиперпараметрический тюнинг и Data Augmentation дают +7 % к mAP.
* YOLOv11 сочетает высокую скорость и конкурентную точность.

Ноутбуки ЛР8:

* `notebook/lab8/my_implementation_yolo11.ipynb`
* `notebook/lab8/ultralitycs_yolo11.ipynb`

Результаты тренировок сохранены в папке `runs/`.

---

## ⚙️ Установка и запуск

1. Клонировать репозиторий и перейти в папку:

   ```bash
   git clone https://github.com/CHISH08/ComputerVisionTasks
   cd ComputerVisionTasks
   ```
2. Создать и активировать окружение:

   ```bash
   conda env create -f environment.yml
   conda activate computer-vision-env
   ```
3. Открыть нужный ноутбук:

   ```bash
   jupyter notebook notebook/lab6/...
   ```

---

## 🧾 Лицензия

Проект распространяется под лицензией MIT.
См. файл `LICENSE`.
