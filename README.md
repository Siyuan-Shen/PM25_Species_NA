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

Version 1.5.0

Siyuan Shen, May 9th, 2024 @ Washington University in St. Louis, MO. USA

Buffer Leave-Cluster-Out (BLCO) CV is a compromise solution for pursuing a Buffer Leave-One-Out(BLOO) CV. Since implementing the real BLOO CV for the whole datasets is computationally difficult, and buffered CV faces the problem of having too less training datasets, the BLCO CV aims to select testing datasets near several seed sites and set buffers around them to achieve reducing the spatial autocorrelation in the training datasets. 

How we implement the BLCO CV is a. Select seeds sites based upon the distribution density of sites; b. calculate the distances from each sites to cluster seeds and find the shortest distances of each sites to a certain seed site; c. determine the criterial radius based upon the number of test sites you want to with held; d. set buffers around selected test sites and exclude all sites within the buffers from training datasets.

Updates:
1. The option for implementing BLCO CV is added.
2. The option for plotting the BLCO testing sites, training sites, and buffers distributions is added.
3. Update the Training_Evaluation_Estimation/NH4/v1.5.0-beta/Evaluation_pkg/iostream.py/AVD_output_text() to have test_beginyears, endbeginyears as input variables.

    Version 1.5.1
    Siyuan Shen, May 21st, 2024 @ Home, St. Louis, MO. USA
    A quality control is added in the Estimation Module. Monthly and annual PWM Species concentrations of different regions, states in USA, and provinces in Canada can be calculated in this module.


Version 1.6.0

Siyuan Shen, Jun 14th, 2024 @ Washington University in St. Louis, MO. USA

Updates:
1. The option for monthly-based training model.
2. Latitude, longitude recording is added.
3. Read previous testing datasets is available now.

    Version 1.6.1 
    Siyuan Shen, Jul 3rd, 2024 @ Washington University in St. Louis, MO. USA
    
    Negative outputs from the model are observed. Add penalty in loss function to constrain th emodel output. Also SHAP values analysis are added.
    Update:
    1. Add the loss function option 'GeoMSE' for regression model. Can constrain the model output larger than -geopysical species. The constraint can be adjuested by setting an appropriate GeoMSE_Lamba1_Penalty1 hypereparameter in the config file.
    2. SHAP Value Analysis is added.
    3. Add Classification structure and Multihead structure.
    
    Version 1.6.2
    Siyuan Shen, Jul 12th, 2024 @ Washington University in St. Louis, MO. USA

    The model inputs are divided by geophysical a prioir values. Settings in Two Combined Models.

    Version 1.6.3
    Siyuan Shen, Jul 17th 2024 @ Washington University in St. Louis, MO. USA

    1. The model can be divided by whichever variables in Two Combined Models Module. 
    2. Add seasonal statistical results.

