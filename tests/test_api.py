"""
Integration Tests for Flask API
Tests API endpoints and request/response handling.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import json
from api.app import app
from api.health_advisor import HealthAdvisor, RiskLevel


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestAPIEndpoints:
    """Test Flask API endpoints."""
    
    def test_home_endpoint(self, client):
        """Test home endpoint."""
        response = client.get('/')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'service' in data
        assert 'endpoints' in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
    
    def test_health_advisory_valid_request(self, client):
        """Test health advisory with valid request."""
        request_data = {
            'aqi': 125.5,
            'user_profile': {
                'age': 65,
                'has_respiratory_condition': True,
                'outdoor_activity_level': 'moderate'
            }
        }
        
        response = client.post(
            '/api/health-advisory',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'risk_level' in data
        assert 'recommendations' in data
        assert isinstance(data['recommendations'], list)
    
    def test_health_advisory_missing_aqi(self, client):
        """Test health advisory with missing AQI."""
        request_data = {'user_profile': {}}
        
        response = client.post(
            '/api/health-advisory',
            data=json.dumps(request_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data


class TestHealthAdvisor:
    """Test health advisory system."""
    
    def test_risk_level_classification(self):
        """Test AQI risk level classification."""
        advisor = HealthAdvisor()
        
        assert advisor.get_risk_level(30) == RiskLevel.GOOD
        assert advisor.get_risk_level(75) == RiskLevel.MODERATE
        assert advisor.get_risk_level(125) == RiskLevel.UNHEALTHY_SENSITIVE
        assert advisor.get_risk_level(175) == RiskLevel.UNHEALTHY
        assert advisor.get_risk_level(250) == RiskLevel.VERY_UNHEALTHY
        assert advisor.get_risk_level(350) == RiskLevel.HAZARDOUS
    
    def test_color_assignment(self):
        """Test risk level color codes."""
        advisor = HealthAdvisor()
        
        good_color = advisor.get_color(RiskLevel.GOOD)
        assert good_color == "#00E400"
        
        hazard_color = advisor.get_color(RiskLevel.HAZARDOUS)
        assert hazard_color == "#7E0023"
    
    def test_advisory_generation_general_public(self):
        """Test advisory for general public."""
        advisor = HealthAdvisor()
        
        advisory = advisor.generate_advisory(75, {'age': 30})
        
        assert advisory['aqi'] == 75
        assert advisory['risk_level'] == 'Moderate'
        assert advisory['is_sensitive_group'] is False
        assert len(advisory['recommendations']) > 0
        assert 'disclaimer' in advisory
    
    def test_advisory_generation_sensitive_group(self):
        """Test advisory for sensitive group."""
        advisor = HealthAdvisor()
        
        advisory = advisor.generate_advisory(
            125,
            {
                'age': 70,
                'has_respiratory_condition': True,
                'outdoor_activity_level': 'high'
            }
        )
        
        assert advisory['is_sensitive_group'] is True
        assert advisory['risk_level'] == 'Unhealthy for Sensitive Groups'
        assert len(advisory['recommendations']) > 0
    
    def test_hazardous_conditions(self):
        """Test advisory for hazardous conditions."""
        advisor = HealthAdvisor()
        
        advisory = advisor.generate_advisory(350, {'age': 45})
        
        assert advisory['risk_level'] == 'Hazardous'
        assert 'avoid all outdoor activities' in ' '.join(advisory['recommendations']).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
