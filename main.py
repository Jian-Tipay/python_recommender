import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import the ENHANCED CF algorithm with research-based weights
from cf_algo_enhanced_weights import EnhancedCollaborativeFilteringEngine
import db

# Create FastAPI app
app = FastAPI(
    title="Boarding House Recommendation System",
    description="Hybrid CF with Research-Based Content Weights (Yaacob et al., 2023; Cervelló-Royo et al., 2021)",
    version="2.0.0"
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
print("Initializing ENHANCED CF engine with research-based weights...")
cf_engine = EnhancedCollaborativeFilteringEngine(database)
print("✓ Service initialized successfully!")
print("\nResearch-Based Weights:")
print("  - Distance/Proximity: 30%")
print("  - Cost/Affordability: 25%")
print("  - Safety & Security: 15%")
print("  - Facilities & Amenities: 10%")
print("  - Room Type/Privacy: 10%")
print("  - Management & Maintenance: 5%")
print("  - Social & Environmental: 5%")


class RecommendationRequest(BaseModel):
    user_id: int
    limit: int = 10


class RecommendationResponse(BaseModel):
    recommendations: List[dict]
    algorithm_used: str
    total_results: int


class PredictionRequest(BaseModel):
    user_id: int
    property_id: int


class PredictionResponse(BaseModel):
    predicted_rating: float
    confidence: str


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Boarding House Recommendation System",
        "version": "2.0.0",
        "algorithm": "Hybrid CF + Research-Based Content Weights",
        "weights": cf_engine.WEIGHTS,
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
            "algorithm": "hybrid_cf_with_research_weights"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/recommendations")
def get_recommendations(user_id: int, limit: int = 10):
    """
    Get personalized property recommendations for a user
    
    Uses hybrid approach with research-based weights:
    - User-based CF (40%): Similar students' preferences
    - Item-based CF (30%): Similar properties
    - Content-based (30%): Research-weighted features
    
    Content weights based on:
    - Distance (30%), Cost (25%), Safety (15%)
    - Facilities (10%), Room Type (10%)
    - Management (5%), Social (5%)
    """
    try:
        recommendations = cf_engine.get_hybrid_recommendations(user_id, limit)
        
        if not recommendations:
            return {
                "recommendations": [],
                "algorithm_used": "none",
                "total_results": 0,
                "message": "No recommendations available"
            }
        
        return {
            "recommendations": recommendations,
            "algorithm_used": "hybrid_cf_with_research_weights",
            "total_results": len(recommendations),
            "weights_info": {
                "user_based_cf": "40%",
                "item_based_cf": "30%",
                "content_based": "30%",
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
def predict_rating(request: PredictionRequest):
    """Predict rating for a user-property pair using hybrid algorithm"""
    try:
        prediction = cf_engine.predict_rating(request.user_id, request.property_id)
        
        # Determine confidence based on number of similar users
        similar_users = cf_engine.find_similar_users(request.user_id, k=5)
        confidence = "high" if len(similar_users) >= 3 else "medium" if len(similar_users) > 0 else "low"
        
        return {
            "predicted_rating": round(prediction, 2),
            "confidence": confidence,
            "algorithm": "hybrid_cf_with_research_weights"
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
            "algorithm": "hybrid_cf_with_research_weights"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/weights")
def get_weights():
    """Get current research-based weights"""
    return {
        "content_weights": cf_engine.WEIGHTS,
        "algorithm_weights": {
            "user_based_cf": 0.40,
            "item_based_cf": 0.30,
            "content_based": 0.30
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


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  Starting Boarding House Recommendation Service")
    print("  Version 2.0 - Research-Based Hybrid CF")
    print("="*70)
    print(f"  Server: http://127.0.0.1:8001")
    print(f"  Docs: http://127.0.0.1:8001/docs")
    print(f"  Weights: http://127.0.0.1:8001/weights")
    print("  Press CTRL+C to stop")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )
