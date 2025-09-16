import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Datos de entrada para el test
test_input = [{
    'maximum': 3.3295140718562872,
    'mean': 0.027873130643010852,
    'std': 0.668453660058961,
    'rms': 0.6688714586546263,
    'skewness': 0.008112507081176003,
    'kurtosis': 4.513121988638369,
    'crest_factor': 4.977808559141244,
    'form_factor': 1.6401277946038735,
    'accelerometer': 'DE',
    'thd': 2.904269116812837,
    'f0': 588
}]



@pytest.mark.parametrize("model_name", ["logistic-regression"])  # Cambia si usas otro run_name
def test_predictfeatures(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"


@pytest.mark.parametrize("model_name", ["logistic-regression"])  # Cambia si usas otro run_name
def test_predictfeatures(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["logistic-regression"])  # Cambia si usas otro run_name
def test_predictfeatures(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["logistic-regression"])  # Cambia si usas otro run_name
def test_predictfeatures(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["logistic-regression"])  # Cambia si usas otro run_name
def test_predictfeatures(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["logistic-regression"])  # Cambia si usas otro run_name
def test_predictfeatures_logistic_regression(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["naive-bayes"])  # Cambia si usas otro run_name
def test_predictfeatures_naive_bayes(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["decision-tree"])  # Cambia si usas otro run_name
def test_predictfeatures_decision_tree(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["random-forest"])  # Cambia si usas otro run_name
def test_predictfeatures_random_forest(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["catboost"])  # Cambia si usas otro run_name
def test_predictfeatures_catboost(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["xgboost"])  # Cambia si usas otro run_name
def test_predictfeatures_xgboost(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["lightgbm"])  # Cambia si usas otro run_name
def test_predictfeatures_lightgbm(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"




@pytest.mark.parametrize("model_name", ["mlp-classifier"])  # Cambia si usas otro run_name
def test_predictfeatures_mlp(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"



@pytest.mark.parametrize("model_name", ["svm"])  # Cambia si usas otro run_name
def test_predictfeatures_svm(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"



@pytest.mark.parametrize("model_name", ["knn"])  # Cambia si usas otro run_name
def test_predictfeatures_knn(model_name):
    response = client.post(
        f"/predictfeatures/{model_name}",
        json=test_input[0]
    )
    
    assert response.status_code == 200, f"Response: {response.text}"
    json_response = response.json()
    
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"


# Datos crudos simulados (2 señales de 2048 elementos cada una)
sample_raw = [[0.1] * 2048, [0.2] * 2048]

# Entrada para el nuevo endpoint
test_raw_input = {
    "accelerometer": "FE",
    "raw": sample_raw
}

# Cambia si usas otro run_name
@pytest.mark.parametrize("model_name", ["logistic-regression"]) 
def test_predictraw(model_name):
    response = client.post(
        f"/predictraw/{model_name}",
        json=test_raw_input
    )

    assert response.status_code == 200, f"Response failed: {response.text}"
    json_response = response.json()

    # Validaciones
    assert "prediction" in json_response, "No se encontró 'prediction' en la respuesta"
    assert isinstance(json_response["prediction"], list), "'prediction' no es una lista"
    assert len(json_response["prediction"]) == len(test_raw_input["raw"]), \
        "El número de predicciones no coincide con el número de señales"
