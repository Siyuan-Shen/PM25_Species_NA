#   For every line of code I wrote down:
#   Please run! Please, please, please! I really love you guys if you run for the first time!
#                          
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#                        No Bug
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                 菩提本无树   明镜亦非台
#                 本来无BUG   何必常修改
#

Siyuan Shen, Nov. 2nd, 2023 @ Washington University in St. Louis, MO. USA


This is version v0.1.0 for PM25 estimation over North America. The whole package include 4 main parts. This version is used to test the AVD spatial CV and add the function of setting learning rate decay in config file.

1. Training Package -  This package is built for training the model. The hyperparamets, model structures, training variables, learning objectives, loss functions and other necessary information for training the model.

2. Evaluation Package - This package is built for evaluation. Including normal spatial cross-validation and buffer leave-one-out (BLOO) spatial cross-validation.

3. Estimation Package - This package is built for estiamtion over North America. Training model for estimation (using all available sites), and predict the map of species.

4. Visualization Package - This package is for all visualization tools, including map plotting, regression line, and other tools. 


Version 1.0.0

Siyuan Shen, Jan. 9th, 2024 @ Washington University in St. Louis, MO. USA

1. Use spatial cross-validation to substitute long-term spatial cross-validation;
2. Add settings of activation function, sturcture of network, loss function, and learning rate in config.toml.
