"""
Health Advisory System
Generates personalized health recommendations based on AQI levels and user risk factors.
"""

from typing import Dict, List
from enum import Enum


class RiskLevel(Enum):
    """AQI risk level categories based on EPA standards."""
    GOOD = "Good"
    MODERATE = "Moderate"
    UNHEALTHY_SENSITIVE = "Unhealthy for Sensitive Groups"
    UNHEALTHY = "Unhealthy"
    VERY_UNHEALTHY = "Very Unhealthy"
    HAZARDOUS = "Hazardous"


class HealthAdvisor:
    """Provides health advisories based on air quality and user profile."""
    
    @staticmethod
    def get_risk_level(aqi: float) -> RiskLevel:
        """
        Determine risk level from AQI value.
        
        Args:
            aqi: Air Quality Index value
            
        Returns:
            Risk level category
        """
        if aqi <= 50:
            return RiskLevel.GOOD
        elif aqi <= 100:
            return RiskLevel.MODERATE
        elif aqi <= 150:
            return RiskLevel.UNHEALTHY_SENSITIVE
        elif aqi <= 200:
            return RiskLevel.UNHEALTHY
        elif aqi <= 300:
            return RiskLevel.VERY_UNHEALTHY
        else:
            return RiskLevel.HAZARDOUS
    
    @staticmethod
    def get_color(risk_level: RiskLevel) -> str:
        """Get color code for risk level."""
        colors = {
            RiskLevel.GOOD: "#00E400",
            RiskLevel.MODERATE: "#FFFF00",
            RiskLevel.UNHEALTHY_SENSITIVE: "#FF7E00",
            RiskLevel.UNHEALTHY: "#FF0000",
            RiskLevel.VERY_UNHEALTHY: "#8F3F97",
            RiskLevel.HAZARDOUS: "#7E0023"
        }
        return colors.get(risk_level, "#808080")
    
    def generate_advisory(
        self,
        aqi: float,
        user_profile: Dict[str, any] = None
    ) -> Dict[str, any]:
        """
        Generate personalized health advisory.
        
        Args:
            aqi: Current or predicted AQI
            user_profile: User risk factors
                - age: int (optional)
                - has_respiratory_condition: bool (default False)
                - has_heart_condition: bool (default False)
                - outdoor_activity_level: str ("low", "moderate", "high")
                
        Returns:
            Dictionary with advisory information
        """
        if user_profile is None:
            user_profile = {}
        
        # Extract user factors
        age = user_profile.get('age', 30)
        has_respiratory = user_profile.get('has_respiratory_condition', False)
        has_heart = user_profile.get('has_heart_condition', False)
        activity_level = user_profile.get('outdoor_activity_level', 'moderate')
        
        # Determine if user is sensitive
        is_sensitive = (
            age < 18 or age > 65 or
            has_respiratory or
            has_heart or
            activity_level == 'high'
        )
        
        # Get risk level
        risk_level = self.get_risk_level(aqi)
        
        # Generate recommendations
        recommendations = self._get_recommendations(risk_level, is_sensitive)
        
        # Create advisory
        advisory = {
            'aqi': round(aqi, 1),
            'risk_level': risk_level.value,
            'color': self.get_color(risk_level),
            'is_sensitive_group': is_sensitive,
            'recommendations': recommendations,
            'health_effects': self._get_health_effects(risk_level, is_sensitive),
            'disclaimer': "This is informational only. Consult healthcare professionals for personalized medical advice."
        }
        
        return advisory
    
    def _get_recommendations(
        self,
        risk_level: RiskLevel,
        is_sensitive: bool
    ) -> List[str]:
        """Get specific recommendations based on risk level."""
        
        recommendations = {
            RiskLevel.GOOD: [
                "Air quality is excellent! Perfect for outdoor activities.",
                "No health concerns for any population group.",
            ],
            RiskLevel.MODERATE: [
                "Air quality is acceptable for most people.",
                "Unusually sensitive individuals may experience minor respiratory symptoms.",
                "Consider reducing prolonged outdoor exertion if you're extremely sensitive.",
            ],
            RiskLevel.UNHEALTHY_SENSITIVE: [
                "Sensitive groups should reduce prolonged outdoor exertion.",
                "Children, elderly, and those with respiratory/heart conditions should limit outdoor activities.",
                "General public can engage in outdoor activities normally.",
            ],
            RiskLevel.UNHEALTHY: [
                "Everyone should reduce prolonged outdoor exertion.",
                "Sensitive groups should avoid prolonged outdoor activities.",
                "Wear a mask (N95/KN95) if you must be outside.",
                "Keep windows and doors closed.",
            ],
            RiskLevel.VERY_UNHEALTHY: [
                "Everyone should avoid prolonged outdoor exertion.",
                "Sensitive groups should remain indoors.",
                "Use air purifiers indoors if available.",
                "Wear N95/KN95 masks when outside is necessary.",
                "Reschedule outdoor activities.",
            ],
            RiskLevel.HAZARDOUS: [
                "Everyone should avoid all outdoor activities.",
                "Remain indoors with windows and doors closed.",
                "Use air purifiers on high settings.",
                "Wear N95/KN95 masks even for brief outdoor exposure.",
                "Seek cleaner air locations if possible (indoor public spaces with filtration).",
            ]
        }
        
        recs = recommendations.get(risk_level, [])
        
        # Add sensitive group warning if applicable
        if is_sensitive and risk_level in [RiskLevel.MODERATE, RiskLevel.UNHEALTHY_SENSITIVE]:
            recs.insert(0, "⚠️ You are in a sensitive group - take extra precautions.")
        
        return recs
    
    def _get_health_effects(
        self,
        risk_level: RiskLevel,
        is_sensitive: bool
    ) -> str:
        """Get health effects description."""
        
        effects = {
            RiskLevel.GOOD: "No health impacts expected.",
            RiskLevel.MODERATE: "Minor respiratory symptoms possible for extremely sensitive individuals.",
            RiskLevel.UNHEALTHY_SENSITIVE: 
                "Increased respiratory symptoms and breathing discomfort for sensitive groups. "
                "General public: minor irritation.",
            RiskLevel.UNHEALTHY:
                "Increased respiratory symptoms for everyone. "
                "Sensitive groups may experience more serious effects.",
            RiskLevel.VERY_UNHEALTHY:
                "Significant respiratory symptoms and reduced lung function. "
                "Increased cardiovascular effects for sensitive groups.",
            RiskLevel.HAZARDOUS:
                "Serious health effects for entire population. "
                "Sensitive groups at severe risk. Emergency conditions."
        }
        
        effect = effects.get(risk_level, "Unknown health effects.")
        
        if is_sensitive:
            effect = "⚠️ SENSITIVE GROUP: " + effect
        
        return effect


def main():
    """Example usage of Health Advisor."""
    advisor = HealthAdvisor()
    
    # Test different AQI levels
    test_cases = [
        (45, {"age": 30, "outdoor_activity_level": "moderate"}),
        (85, {"age": 70, "has_respiratory_condition": True}),
        (165, {"age": 10}),
        (225, {"has_heart_condition": True}),
    ]
    
    print("="*70)
    print(" HEALTH ADVISORY SYSTEM - TEST CASES ")
    print("="*70)
    
    for aqi, profile in test_cases:
        advisory = advisor.generate_advisory(aqi, profile)
        
        print(f"\n📊 AQI: {advisory['aqi']} - {advisory['risk_level']}")
        print(f"   Color: {advisory['color']}")
        print(f"   Sensitive Group: {'Yes' if advisory['is_sensitive_group'] else 'No'}")
        print(f"\n   Health Effects:")
        print(f"   {advisory['health_effects']}")
        print(f"\n   Recommendations:")
        for i, rec in enumerate(advisory['recommendations'], 1):
            print(f"   {i}. {rec}")
        print("\n" + "-"*70)


if __name__ == "__main__":
    main()
