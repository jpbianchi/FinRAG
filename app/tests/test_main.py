import os, sys
sys.path.append("../")
from main import app

from fastapi.testclient import TestClient

from settings import datadir

client = TestClient(app)

def test_read_root():
    response = client.get("/ping/")
    assert response.status_code == 200
    assert int(response.json()['answer']) < 100
    
def test_list_files():
    response = client.get("/list_files/")
    files = os.listdir(datadir)
    assert response.status_code == 200
    assert len(response.json()['files']) == len(files)
    for f in response.json()['files']:
        assert f in files
        
def test_vector_search():
    question_data = {"question": "Does ATT have postpaid phone customers?"}
    response = client.post("/ask/", json=question_data)
    assert response.status_code == 200
    assert len(response.json()['answer']) > 0 # we assume vector store works if it returns something
    assert any(['postpaid' in a.lower() for a in response.json()['answer']])
    

def test_ragit():
    question_data = {"question": "Does ATT have postpaid phone customers?"}
    response = client.post("/ragit/", json=question_data)
    assert response.status_code == 200
    assert 'yes' in response.json()['answer'].lower()
