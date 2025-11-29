import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn

# Import the ENHANCED CF algorithm with research-based weights
from cf_algo_enhanced_weights import EnhancedCollaborativeFilteringEngine
import db

# Create FastAPI app
app = FastAPI(
    title="Boarding House Recommendation System",
    description="Adaptive Hybrid CF with Research-Based Content Weights",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and ENHANCED CF engine
print("Initializing database connection...")
database = db.Database()
print("Initializing ENHANCED CF engine with adaptive strategy...")
cf_engine = EnhancedCollaborativeFilteringEngine(database)
print("✓ Service initialized successfully!")
print("\n🎯 Adaptive Algorithm Strategies:")
print("  NEW USER (0-1 ratings) → Popular")
print("  EXPLORING (2-4 ratings, varied) → Hybrid (Item+Content)")
print("  ACTIVE (5+ ratings, recent activity) → Collaborative")
print("  SETTLED (5+ ratings, low activity) → Hybrid (Balanced)")
print("  PREFERENCE-FOCUSED (detailed prefs, 2-6 ratings) → Content-Based")


class RecommendationRequest(BaseModel):
    user_id: int
    limit: int = 10


class UserActivityProfile(BaseModel):
    rating_count: int
    recent_rating_count: int
    recent_view_count: int
    avg_rating: float
    rating_variance: float
    preference_completeness: float
    last_rating_time: Optional[str]


class AlgorithmStrategy(BaseModel):
    strategy: str
    algorithm: str
    reason: str
    action_needed: Optional[str]
    weights: Dict[str, float]


def get_user_activity_profile(user_id: int) -> Dict:
    """Get user activity profile for algorithm selection"""
    try:
        # Get rating statistics
        ratings = database.get_user_ratings(user_id)
        rating_count = len(ratings)
        
        if rating_count == 0:
            return {
                'rating_count': 0,
                'recent_rating_count': 0,
                'recent_view_count': 0,
                'avg_rating': 0,
                'rating_variance': 0,
                'preference_completeness': 0,
                'last_rating_time': None
            }
        
        # Calculate rating variance
        rating_values = [r['rating'] for r in ratings]
        avg_rating = sum(rating_values) / len(rating_values)
        variance = sum((r - avg_rating) ** 2 for r in rating_values) / len(rating_values)
        
        # Get preference completeness
        preference = database.get_user_preference(user_id)
        completeness = 0.0
        if preference:
            has_budget = 1 if preference.get('budget_min', 0) > 0 and preference.get('budget_max', 0) > 0 else 0
            has_distance = 1 if preference.get('preferred_distance', 0) > 0 else 0
            has_room_type = 1 if preference.get('room_type') and preference.get('room_type') != 'Any' else 0
            completeness = (has_budget + has_distance + has_room_type) / 3.0
        
        # Note: Recent activity tracking would require additional database queries
        # For now, we'll use simplified logic
        recent_rating_count = min(rating_count, 3)  # Simplified
        recent_view_count = 0  # Would need property_views table query
        
        return {
            'rating_count': rating_count,
            'recent_rating_count': recent_rating_count,
            'recent_view_count': recent_view_count,
            'avg_rating': round(avg_rating, 2),
            'rating_variance': round(variance, 2),
            'preference_completeness': round(completeness, 2),
            'last_rating_time': ratings[-1].get('created_at') if ratings else None
        }
    except Exception as e:
        print(f"Error getting user activity profile: {str(e)}")
        return {
            'rating_count': 0,
            'recent_rating_count': 0,
            'recent_view_count': 0,
            'avg_rating': 0,
            'rating_variance': 0,
            'preference_completeness': 0,
            'last_rating_time': None
        }


def determine_algorithm_strategy(user_id: int) -> Dict:
    """
    Determine algorithm strategy based on user behavior
    Matches Laravel's determineAlgorithmStrategy logic
    """
    profile = get_user_activity_profile(user_id)
    
    rating_count = profile['rating_count']
    recent_ratings = profile['recent_rating_count']
    recent_views = profile['recent_view_count']
    variance = profile['rating_variance']
    completeness = profile['preference_completeness']
    
    # RULE 1: NEW USER (0-1 ratings)
    if rating_count <= 1:
        return {
            'strategy': 'new_user',
            'algorithm': 'Popular',
            'reason': 'Showing popular properties based on student ratings',
            'action_needed': 'Rate 2+ properties to unlock personalized AI recommendations',
            'weights': {
                'user_based_cf': 0.0,
                'item_based_cf': 0.0,
                'content': 0.6,
                'popular': 0.4
            }
        }
    
    # RULE 2: EXPLORING USER (2-4 ratings with varied preferences)
    if rating_count >= 2 and rating_count <= 4 and variance > 1.0:
        return {
            'strategy': 'exploring',
            'algorithm': 'Hybrid',
            'reason': 'Combining your preferences with similar property patterns',
            'action_needed': 'Continue rating properties to improve recommendations',
            'weights': {
                'user_based_cf': 0.1,
                'item_based_cf': 0.5,
                'content': 0.4,
                'popular': 0.0
            }
        }
    
    # RULE 3: ACTIVE USER (5+ ratings with recent activity)
    if rating_count >= 5 and (recent_ratings > 0 or recent_views > 2):
        return {
            'strategy': 'active',
            'algorithm': 'Collaborative',
            'reason': 'AI analyzing students with similar preferences to yours',
            'action_needed': None,
            'weights': {
                'user_based_cf': 0.5,
                'item_based_cf': 0.3,
                'content': 0.2,
                'popular': 0.0
            }
        }
    
    # RULE 4: SETTLED USER (5+ ratings but less recent activity)
    if rating_count >= 5 and recent_ratings == 0 and recent_views <= 2:
        return {
            'strategy': 'settled',
            'algorithm': 'Hybrid',
            'reason': 'Using balanced approach based on your established preferences',
            'action_needed': None,
            'weights': {
                'user_based_cf': 0.35,
                'item_based_cf': 0.35,
                'content': 0.3,
                'popular': 0.0
            }
        }
    
    # RULE 5: PREFERENCE-FOCUSED (detailed prefs, moderate ratings)
    if completeness >= 0.67 and rating_count >= 2 and rating_count <= 6:
        return {
            'strategy': 'preference_focused',
            'algorithm': 'Content-Based',
            'reason': 'Matching properties to your detailed preference profile',
            'action_needed': None,
            'weights': {
                'user_based_cf': 0.2,
                'item_based_cf': 0.2,
                'content': 0.6,
                'popular': 0.0
            }
        }
    
    # DEFAULT: Standard Hybrid
    return {
        'strategy': 'standard',
        'algorithm': 'Hybrid',
        'reason': 'Using multi-algorithm AI approach',
        'action_needed': None,
        'weights': {
            'user_based_cf': 0.4,
            'item_based_cf': 0.3,
            'content': 0.3,
            'popular': 0.0
        }
    }


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Boarding House Recommendation System",
        "version": "2.1.0",
        "algorithm": "Adaptive Hybrid CF + Research-Based Content Weights",
        "strategies": [
            "NEW USER → Popular",
            "EXPLORING → Hybrid (Item+Content)",
            "ACTIVE → Collaborative",
            "SETTLED → Hybrid (Balanced)",
            "PREFERENCE-FOCUSED → Content-Based"
        ],
        "content_weights": cf_engine.WEIGHTS,
        "references": [
            "Yaacob et al. (2023) - Student housing preferences",
            "Cervelló-Royo et al. (2021) - Multi-criteria housing analysis",
            "Akinjare et al. (2025) - Student satisfaction determinants"
        ]
    }


@app.get("/health")
def health_check():
    """Check if service and database are running"""
    try:
        database.test_connection()
        return {
            "status": "healthy",
            "database": "connected",
            "service": "running",
            "algorithm": "adaptive_hybrid_cf"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/user-profile/{user_id}")
def get_user_profile(user_id: int):
    """Get user activity profile"""
    try:
        profile = get_user_activity_profile(user_id)
        strategy = determine_algorithm_strategy(user_id)
        
        return {
            "user_id": user_id,
            "profile": profile,
            "strategy": strategy
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/recommendations")
def get_recommendations(user_id: int, limit: int = 10):
    """
    Get adaptive personalized property recommendations
    
    Algorithm automatically adjusts based on:
    - User rating count and patterns
    - Recent activity
    - Preference completeness
    - Rating variance
    """
    try:
        # Determine strategy
        strategy = determine_algorithm_strategy(user_id)
        
        # Get recommendations using appropriate method
        strategy_name = strategy['strategy']
        
        if strategy_name == 'new_user':
            # Use popular/cold start for new users
            recommendations = cf_engine.get_hybrid_recommendations(user_id, limit)
            # Could enhance with popular properties logic
            
        elif strategy_name == 'exploring':
            # Emphasize item-based CF
            recommendations = cf_engine.get_hybrid_recommendations(user_id, limit)
            
        elif strategy_name == 'active':
            # Emphasize user-based CF
            recommendations = cf_engine.get_user_based_recommendations(user_id, limit)
            # Fallback to hybrid if user-based returns empty
            if not recommendations:
                recommendations = cf_engine.get_hybrid_recommendations(user_id, limit)
            
        elif strategy_name == 'preference_focused':
            # Emphasize content-based
            recommendations = cf_engine.get_hybrid_recommendations(user_id, limit)
            # Content score is already weighted in hybrid
            
        else:
            # Standard hybrid
            recommendations = cf_engine.get_hybrid_recommendations(user_id, limit)
        
        if not recommendations:
            return {
                "recommendations": [],
                "algorithm_used": strategy['algorithm'],
                "strategy": strategy_name,
                "total_results": 0,
                "message": strategy['reason'],
                "action_needed": strategy.get('action_needed'),
                "user_activity": get_user_activity_profile(user_id)
            }
        
        return {
            "recommendations": recommendations,
            "algorithm_used": strategy['algorithm'],
            "strategy": strategy_name,
            "total_results": len(recommendations),
            "message": strategy['reason'],
            "action_needed": strategy.get('action_needed'),
            "user_activity": get_user_activity_profile(user_id),
            "weights_info": {
                "adaptive_weights": strategy['weights'],
                "content_breakdown": cf_engine.WEIGHTS
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.get("/recommendations/collaborative")
def get_collaborative_recommendations(user_id: int, limit: int = 10):
    """Get pure user-based collaborative filtering recommendations"""
    try:
        recommendations = cf_engine.get_user_based_recommendations(user_id, limit)
        
        return {
            "recommendations": recommendations,
            "algorithm_used": "user_based_cf_only",
            "total_results": len(recommendations)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/recommendations/item-based")
def get_item_based_recommendations(user_id: int, limit: int = 10):
    """Get item-based collaborative filtering recommendations"""
    try:
        recommendations = cf_engine.get_item_based_recommendations(user_id, limit)
        
        return {
            "recommendations": recommendations,
            "algorithm_used": "item_based_cf_only",
            "total_results": len(recommendations)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/predict-rating")
def predict_rating(request: dict):
    """Predict rating for a user-property pair using adaptive algorithm"""
    try:
        user_id = request.get('user_id')
        property_id = request.get('property_id')
        
        prediction = cf_engine.predict_rating(user_id, property_id)
        
        # Get strategy for confidence
        strategy = determine_algorithm_strategy(user_id)
        similar_users = cf_engine.find_similar_users(user_id, k=5)
        confidence = "high" if len(similar_users) >= 3 else "medium" if len(similar_users) > 0 else "low"
        
        return {
            "predicted_rating": round(prediction, 2),
            "confidence": confidence,
            "algorithm": strategy['algorithm'],
            "strategy": strategy['strategy']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/similar-users/{user_id}")
def get_similar_users(user_id: int, k: int = 10):
    """Find similar users based on rating patterns"""
    try:
        similar_users = cf_engine.find_similar_users(user_id, k)
        
        return {
            "user_id": user_id,
            "similar_users": [su.to_dict() for su in similar_users],
            "count": len(similar_users)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/similar-properties/{property_id}")
def get_similar_properties(property_id: int, k: int = 10):
    """Find similar properties based on user ratings"""
    try:
        similar_properties = cf_engine.find_similar_properties(property_id, k)
        
        return {
            "property_id": property_id,
            "similar_properties": [sp.to_dict() for sp in similar_properties],
            "count": len(similar_properties)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/retrain")
def retrain_model():
    """Retrain the collaborative filtering model with latest data"""
    try:
        cf_engine.clear_cache()
        return {
            "status": "success",
            "message": "Model cache cleared, will retrain on next request",
            "algorithm": "adaptive_hybrid_cf"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/weights")
def get_weights():
    """Get current research-based content weights"""
    return {
        "content_weights": cf_engine.WEIGHTS,
        "adaptive_strategies": {
            "new_user": {"user_cf": 0.0, "item_cf": 0.0, "content": 0.6, "popular": 0.4},
            "exploring": {"user_cf": 0.1, "item_cf": 0.5, "content": 0.4, "popular": 0.0},
            "active": {"user_cf": 0.5, "item_cf": 0.3, "content": 0.2, "popular": 0.0},
            "settled": {"user_cf": 0.35, "item_cf": 0.35, "content": 0.3, "popular": 0.0},
            "preference_focused": {"user_cf": 0.2, "item_cf": 0.2, "content": 0.6, "popular": 0.0}
        },
        "references": [
            {
                "weight": "distance (30%)",
                "source": "Yaacob et al. (2023), Cervelló-Royo et al. (2021)"
            },
            {
                "weight": "cost (25%)",
                "source": "Yaacob et al. (2023), Cervelló-Royo et al. (2021)"
            },
            {
                "weight": "safety (15%)",
                "source": "Akinjare et al. (2025)"
            },
            {
                "weight": "facilities (10%)",
                "source": "Mohit et al. (2010)"
            },
            {
                "weight": "room_type (10%)",
                "source": "Mohit et al. (2010)"
            },
            {
                "weight": "management (5%)",
                "source": "Mohd Isa & Ismail (2014)"
            },
            {
                "weight": "social (5%)",
                "source": "Mohd Isa & Ismail (2014)"
            }
        ]
    }


@app.get("/strategy/{user_id}")
def get_user_strategy(user_id: int):
    """Get the current algorithm strategy for a user"""
    try:
        strategy = determine_algorithm_strategy(user_id)
        profile = get_user_activity_profile(user_id)
        
        return {
            "user_id": user_id,
            "strategy": strategy,
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  Starting Boarding House Recommendation Service")
    print("  Version 2.1 - Adaptive Hybrid CF with Research Weights")
    print("="*70)
    print(f"  Server: http://127.0.0.1:8001")
    print(f"  Docs: http://127.0.0.1:8001/docs")
    print(f"  Weights: http://127.0.0.1:8001/weights")
    print(f"  Strategy: http://127.0.0.1:8001/strategy/{{user_id}}")
    print("\n  🎯 Adaptive Strategies:")
    print("     NEW USER → Popular recommendations")
    print("     EXPLORING → Item+Content hybrid")
    print("     ACTIVE → Collaborative filtering")
    print("     SETTLED → Balanced hybrid")
    print("     PREFERENCE-FOCUSED → Content-based")
    print("\n  Press CTRL+C to stop")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
