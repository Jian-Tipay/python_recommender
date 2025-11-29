import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now we can import everything
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# Import the ENHANCED CF algorithm
from cf_algo_enhanced import EnhancedCollaborativeFilteringEngine
import db

# Create FastAPI app
app = FastAPI(
    title="Boarding House Recommendation System",
    description="AI-powered collaborative filtering recommendation service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database and CF engine
database = None
cf_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global database, cf_engine
    try:
        print("="*60)
        print("🚀 Starting Boarding House CF Recommendation Service")
        print("="*60)
        
        print("📊 Initializing database connection...")
        database = db.Database()
        database.test_connection()
        print("✅ Database connected successfully!")
        
        print("🤖 Initializing Enhanced CF engine...")
        cf_engine = EnhancedCollaborativeFilteringEngine(database)
        print("✅ CF Engine initialized!")
        
        # Get some stats
        stats = database.get_rating_statistics()
        if stats:
            print(f"📈 System Stats:")
            print(f"   - Users: {stats.get('total_users', 0)}")
            print(f"   - Properties: {stats.get('total_properties', 0)}")
            print(f"   - Ratings: {stats.get('total_ratings', 0)}")
        
        print("="*60)
        print("✅ Service ready!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        raise


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
    """Root endpoint"""
    return {
        "status": "online",
        "service": "Boarding House Recommendation System",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommendations": "/recommendations?user_id=X&limit=10",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """Check if service and database are running"""
    try:
        if database is None:
            raise Exception("Database not initialized")
        
        # Test database connection
        database.test_connection()
        
        # Get system stats
        stats = database.get_rating_statistics()
        
        return {
            "status": "healthy",
            "database": "connected",
            "service": "running",
            "stats": {
                "total_users": stats.get('total_users', 0) if stats else 0,
                "total_properties": stats.get('total_properties', 0) if stats else 0,
                "total_ratings": stats.get('total_ratings', 0) if stats else 0
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Service unhealthy: {str(e)}"
        )


@app.get("/recommendations")
def get_recommendations(user_id: int, limit: int = 10):
    """
    Get personalized property recommendations for a user
    Uses hybrid approach: User-based CF + Item-based CF + Content filtering
    """
    try:
        if cf_engine is None:
            raise HTTPException(
                status_code=503, 
                detail="CF Engine not initialized"
            )
        
        recommendations = cf_engine.get_hybrid_recommendations(user_id, limit)
        
        if not recommendations:
            return {
                "recommendations": [],
                "algorithm_used": "none",
                "total_results": 0,
                "message": "No recommendations available. User may need to rate more properties."
            }
        
        return {
            "recommendations": recommendations,
            "algorithm_used": "hybrid",
            "total_results": len(recommendations)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating recommendations: {str(e)}"
        )


@app.get("/recommendations/collaborative")
def get_collaborative_recommendations(user_id: int, limit: int = 10):
    """Get pure user-based collaborative filtering recommendations"""
    try:
        if cf_engine is None:
            raise HTTPException(
                status_code=503, 
                detail="CF Engine not initialized"
            )
        
        recommendations = cf_engine.get_user_based_recommendations(user_id, limit)
        
        return {
            "recommendations": recommendations,
            "algorithm_used": "user_based_cf",
            "total_results": len(recommendations)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )


@app.get("/recommendations/item-based")
def get_item_based_recommendations(user_id: int, limit: int = 10):
    """Get item-based collaborative filtering recommendations"""
    try:
        if cf_engine is None:
            raise HTTPException(
                status_code=503, 
                detail="CF Engine not initialized"
            )
        
        recommendations = cf_engine.get_item_based_recommendations(user_id, limit)
        
        return {
            "recommendations": recommendations,
            "algorithm_used": "item_based_cf",
            "total_results": len(recommendations)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )


@app.post("/predict-rating")
def predict_rating(request: PredictionRequest):
    """Predict rating for a user-property pair"""
    try:
        if cf_engine is None:
            raise HTTPException(
                status_code=503, 
                detail="CF Engine not initialized"
            )
        
        prediction = cf_engine.predict_rating(request.user_id, request.property_id)
        
        # Determine confidence based on number of similar users
        similar_users = cf_engine.find_similar_users(request.user_id, k=5)
        confidence = "high" if len(similar_users) >= 3 else "medium" if len(similar_users) > 0 else "low"
        
        return {
            "predicted_rating": round(prediction, 2),
            "confidence": confidence
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )


@app.get("/similar-users/{user_id}")
def get_similar_users(user_id: int, k: int = 10):
    """Find similar users based on rating patterns"""
    try:
        if cf_engine is None:
            raise HTTPException(
                status_code=503, 
                detail="CF Engine not initialized"
            )
        
        similar_users = cf_engine.find_similar_users(user_id, k)
        
        return {
            "user_id": user_id,
            "similar_users": [su.to_dict() for su in similar_users],
            "count": len(similar_users)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )


@app.get("/similar-properties/{property_id}")
def get_similar_properties(property_id: int, k: int = 10):
    """Find similar properties based on user ratings"""
    try:
        if cf_engine is None:
            raise HTTPException(
                status_code=503, 
                detail="CF Engine not initialized"
            )
        
        similar_properties = cf_engine.find_similar_properties(property_id, k)
        
        return {
            "property_id": property_id,
            "similar_properties": [sp.to_dict() for sp in similar_properties],
            "count": len(similar_properties)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )


@app.post("/retrain")
def retrain_model():
    """Retrain the collaborative filtering model with latest data"""
    try:
        if cf_engine is None:
            raise HTTPException(
                status_code=503, 
                detail="CF Engine not initialized"
            )
        
        cf_engine.clear_cache()
        return {
            "status": "success",
            "message": "Model cache cleared, will retrain on next request"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Starting Boarding House CF Recommendation Service")
    print("="*60)
    print(f"  Server: http://0.0.0.0:8001")
    print(f"  Docs: http://0.0.0.0:8001/docs")
    print("  Press CTRL+C to stop")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
