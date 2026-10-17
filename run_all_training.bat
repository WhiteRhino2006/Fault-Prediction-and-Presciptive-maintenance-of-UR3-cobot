@echo off
REM Re-runs every offline experiment in the correct order (train.py rebuilds the results table,
REM the other two then add their rows). Takes about 45 minutes. Not needed for the live demo.
cd /d "%~dp0"
echo [1/6] Main models, ablation, root-cause experiment, figures ...
python train.py || goto :error
echo [2/6] Graph neural networks (GAT, Causal-GAT) ...
python train_gnn.py || goto :error
echo [3/6] Deep time-series baselines (LSTM, GRU) ...
python train_lstm.py || goto :error
echo [4/6] Extra baselines (SVM, 1D-CNN, Transformer) ...
python train_baselines_extra.py || goto :error
echo [5/6] Root-cause benchmark (7 scenarios x 5 methods) ...
python exp_root_cause.py || goto :error
echo [6/6] Closed-loop control ablation ...
python exp_closed_loop.py || goto :error
echo.
echo All done. Results: outputs\results.csv and outputs\figures
pause
exit /b
:error
echo.
echo A step failed - see the message above.
pause
