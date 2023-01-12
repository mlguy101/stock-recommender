"""

1- Get Hopt calibration data
2- calibrate find-peak parameters based on backtestpy optimize
3- use these parameters to detect local maxima, minima
4- for each local minima / maxima , calculate the slope for next maxima / minima ( later idea)
5- this normalized slope presents the by which we recommend buy sell
6 - brainstorm limitations
7- create model to predict local maxima / minima from set of indicators and past data only (lagged indicators ?? )
8- iterate and backtest
9- test live with fake money
10  -think about adding risk factor , this will control buy sell based on the accuracy  / confidence of the prediction
"""