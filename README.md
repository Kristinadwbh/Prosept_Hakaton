В основном файле представленны функции по обучению модели и предсказанию. В функции обучения модели данные 
векторизовались с помощью метода tf-idf. До этого данные подверглись изменениям. В названии товаров изначально были
неаккуратно склеены слова. Так были склеены между собой слова на латинице и кириллице, числа и слова, слова в разных
регистрах. С помощью созданной функции их получилось разделить. Затем удалили знаки препинания, произвелась
токкенизация, лемматизация и стемм. Затем строилась матрица евклидовых расстояний с помощью инструмента
scipy.spatial.distance.cdist, которые затем сортировались. 
Данные при этом были почищены. Из данных производителя были удалены 2 строки с пропусками в названии, а из данных 
от диллеров только для обучения были удаленны товары с отсутствующим ключом id к данным производителя. Для тестирования 
и предсказания данные от диллера не удалялись.
К матрице остортированных расстояний добавился target. На месте верных id в матрице расстояний проставили 1,в остальных
случаях 0. Таким образом, предсказываемой переменной стал номер столбца верного id или номер столбца при котором стоит 1.
Матрица расстояний подавалась на вход модели LGBM с мультиклассовой классификацией. Таким образом получились вероятности, 
которые были отсортированны для каждой карточки диллера. Были выбраны 5 карточек наиболее вероятных карточек производителя
для каждой карточки диллера. На данных, в которых были удаленны неразмеченные карточки диллеров, метрика вышла 97%. На
изначальных данных - 84%. Тест производился на отдельной выборке.
В общую функцию для обучения подаются три или четыре списка словарей. Данные производителя, данные диллеров,
на которых модель может обучиться (в нашем случае, те данные от диллеров, которые были даны), данные с внешними ключами и 
данные для предсказания. Последний список по умолчанию является тем же списком от диллеров, что и для обучения, но можно
ввести другие данные для предсказания.
Функция возвращает список словарей, в котором каждый список (каждая карточка от диллера) представлен 5 id.
