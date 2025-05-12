import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mlflow.sklearn

def treinar_modelo(path_dados):
    df = pd.read_csv(path_dados)
    X = df[['temperatura']]
    y = df['vendas_sorvete']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    score = modelo.score(X_test, y_test)

    with mlflow.start_run():
        mlflow.log_param("modelo", "LinearRegression")
        mlflow.log_metric("r2_score", score)
        mlflow.sklearn.log_model(modelo, "modelo_icecream")

    return modelo
