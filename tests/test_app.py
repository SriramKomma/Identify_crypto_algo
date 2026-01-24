def test_index_page(client):
    response = client.get('/')
    assert response.status_code == 200
    # Check for some content we expect on the page
    assert b"Cipher Algorithm Identifier" in response.data or b"Algorithm" in response.data

def test_api_stats(client):
    response = client.get('/api/stats')
    assert response.status_code == 200
    assert response.json['labels'] is not None

def test_prediction_flow(client):
    # Test with a dummy ciphertext "Hello World" in hex
    response = client.post('/predict', data={
        'ciphertext': '48656c6c6f20576f726c64'
    }, follow_redirects=True)
    
    assert response.status_code == 200
    # Should show the result section
    assert b"Detected:" in response.data
