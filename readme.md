В этой задаче используется полносвязная нейронная сеть из пяти слоев: один Flatten и четыре Dense. В первых трех слоях Dense используется активация selu, в последнем softmax. В слоях Dens 256, 256, 64 и 10 нейронов соответсвенно. 

В цикле обучения используется оптимизатор SGB, функция потерь sparse_categorical_crossentropy и метрика accuracy. Всего производится 15 эпох обучения.

Функции потерь 

![Image alt](https://github.com/samsdimko/SMOMI/Loss.png)

Метрики точности

![Image alt](https://github.com/samsdimko/SMOMI/Acc.png)

Итоговая точность 0.4963