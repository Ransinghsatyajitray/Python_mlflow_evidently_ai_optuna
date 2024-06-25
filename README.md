# Python_mlflow_evidently_ai_optuna
This project is for ML using sklearn which covers classification, mlflow, evidentlyai, optuna, fastapi, flask, docker



# Optuna

Randomsearchcv, gridsearchcv work well when we have smaller no of parameters to be optimized. When the number of parameters gets increased, these optimization techniques become computation extensive.

# Optuna uses the bayesian optimization technique which help us to select better parameters to be selected.
1. Build a surrogate probability model of the objective function
2. Find the hyperparameters that perform best on the surrogate
3. Apply these hyperparameters to the true objective functions
4. Update the surrogate model incorporating the new results
5. Repeat steps 2-4 until max iterations or time is reached.

We always define an objective function

For running the application locally outside the docker container, 
python app.py

see using:
http://localhost:5000/predict



building the docker : docker build -t ml_model2 .
running the docker container: docker run -p 5000:5000 ml_model2