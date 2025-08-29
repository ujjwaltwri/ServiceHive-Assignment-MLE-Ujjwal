import pytest
from fastapi.testclient import TestClient
from SIC.api.main import app

# This is a pytest fixture. It creates a TestClient that correctly handles
# the lifespan events (model loading) before the tests run.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """
    Tests if the /health endpoint returns a 200 OK status and confirms models are loaded.
    """
    response = client.get("/health")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "ok"
    # This will now pass because the fixture ensures models are loaded.
    assert json_response["models_loaded"] is True

def test_predict_success(client):
    """
    Tests a successful prediction with a valid JPG image.
    """
    # Ensure you have renamed your test image to "test_image.jpg"
    file_path = "tests/assets/test_image.jpg"
    
    with open(file_path, "rb") as f:
        response = client.post("/predict", files={"file": ("test_image.jpg", f, "image/jpeg")})
            
    assert response.status_code == 200
    response_data = response.json()
    assert "prediction" in response_data
    assert "confidence" in response_data
    assert "uncertainty" in response_data
    assert "description" in response_data

def test_predict_invalid_file(client):
    """
    Tests if the API correctly handles a non-image file upload.
    """
    invalid_file_content = b"this is not an image"
    
    response = client.post("/predict", files={"file": ("test.txt", invalid_file_content, "text/plain")})
            
    # This assertion will now pass because the models are loaded, allowing the
    # endpoint to correctly identify the invalid file and return a 400 error.
    assert response.status_code == 400
    assert "Invalid image file" in response.json()["detail"]