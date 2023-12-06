<p align="center">
  <img src="https://prosept.ru/images/prosept-logo.svg" />
</p>

# Описание проекта

Заказчик производит несколько сотен различных товаров бытовой и промышленной
химии, а затем продаёт эти товары через дилеров. Дилеры, в свою очередь,
занимаются розничной продажей товаров в крупных сетях магазинов и на онлайн
площадках.Для оценки ситуации, управления ценами и бизнесом в целом, заказчик
периодически собирает информацию о том, как дилеры продают их товар. Для этого
они парсят сайты дилеров, а затем сопоставляют товары и цены.
Зачастую описание товаров на сайтах дилеров отличаются от того описания, что
даёт заказчик.

Цель этого проекта - разработка решения, которое отчасти автоматизирует процесс
сопоставления товаров. Основная идея - предлагать несколько товаров заказчика,
которые с наибольшей вероятностью соответствуют размечаемому товару дилера.
Предлагается реализовать это решение, как онлайн сервис, открываемый в веб-
браузере. Выбор наиболее вероятных подсказок делается методами машинного
обучения.

# Работа Приложения

В основном файле **main.py** представленны функции по обучению модели и предсказанию. Данные
векторизовались с помощью трансформера **LaBSE**. До этого данные подверглись
изменениям. В названии товаров изначально были
неаккуратно склеены слова. Так были склеены между собой слова на латинице и
кириллице, числа и слова, слова в разных
регистрах. С помощью созданной функции их получилось разделить. Затем удалили
знаки препинания, стоп-слова, произвелась
токкенизация, лемматизация и стемм. Затем строилась матрица евклидовых
расстояний с помощью инструмента
**scipy.spatial.distance.cdist**. Расстояния затем сортировались.
Данные при этом были почищены. Из данных производителя были удалены 2 строки с
пропусками в названии. Только лишь для обучения из данных диллеров были удалены строки с удалёнными id из данных призводителя, а также товары с отсутствующим ключом id к
данным производителя. 

Для тестирования
и предсказания данные от диллера не удалялись.
К матрице остортированных расстояний добавился **target**. На месте верных id в
матрице расстояний проставили 1, в остальных
случаях 0. Таким образом, предсказываемой переменной стал номер столбца верного
id или номер столбца при котором стоит 1.
Матрица расстояний подавалась на вход модели **LGBM** с мультиклассовой
классификацией. Таким образом получились вероятности,
которые были отсортированны для каждой карточки диллера. 

Были выбраны 5 карточек
наиболее вероятных карточек производителя
для каждой карточки диллера. В функции предсказания прописан выбор любого колличества выдаваемых id,
но по умолчанию 5. Тест производился на отдельной выборке. Так для 30% данных диллера, на которых производился тест, метрика для 5 карточек id вышла примерно **0.985**.
На вход в функцию для обучения подаются четыре параметра. Список словарей данных производителя, список словарей дынных диллера для обучения, список словарей данных внешних ключей и название столбца из данных производителя, по которому смотрится описание и ищутся соответствия, по умолчанию это столбец **name**, но можно подставить к примеру и **name_1c**. Функция обучения возвращает два объекта - обученную модель и фрэйм с эмбеддингами данных производителя. 
В функцию для предсказания на вход подаются пять параметров. Список словарей данных производителя, список словарей данных диллеров для предсказания, кортеж из двух объектов - обученной модели и фрэйма эмбеддингов данных производителя (которые были возвращены функцией обучения), параметр **k** - колличество выдаваемых id (по умолчанию 5), и параметр **nm** - название столбца с названием товаров из данных производителя, по которому находятся соотвтествия с названиями из данных диллеров ( по умолчанию - **name**, но можно к примеру и **name_1c**, при условии что такое же название стоит в параметре nm функции обучения). Функция предсказания возвращает список словарей id (по k штук самых вероятных id из данных производителя для каждой карточки диллера, в порядка убывания вероятности). 

В файле **example.py** показан пример использования программы и подсчёта метрики.
В файле **realization.ipynb** представленно выполнение работы над программой. С примерами предсказания для 30% данных диллера с 5 вариантами id и для 30 карточек диллеров с 3 вариантами id.


# Авторы:

- **[Kristina](https://t.me/kr1588)**
- **[Julia](https://t.me/Jshmatova)**
- **[Alexandr](https://t.me/AlexXramov)**
