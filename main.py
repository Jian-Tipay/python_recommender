import sys
import os
from datetime import datetime, timedelta
from threading import Lock

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

from cf_algo_enhanced_weights import EnhancedCollaborativeFilteringEngine
import db

# ============================================================================
# SMART DATA MANAGER - Auto-detects when to refresh data
# ============================================================================
class SmartDataManager:
    """
    Manages data freshness intelligently:
    - Refreshes automatically every N minutes
    - Force refreshes on demand
    - Per-user cache invalidation
    - Thread-safe operations
    """
    def __init__(self, auto_refresh_minutes=5):
        self.database = None
        self.cf_engine = None
        self.last_full_refresh = None
        self.auto_refresh_interval = timedelta(minutes=auto_refresh_minutes)
        self.user_cache_invalidated = set()  # Track users with stale cache
        self.lock = Lock()

        # Initial load
        self._full_refresh()

    def _full_refresh(self):
        """Complete data reload"""
        with self.lock:
            logger.info("🔄 Performing full data refresh...")
            self.database = db.Database()
            self.cf_engine = EnhancedCollaborativeFilteringEngine(self.database)
            self.last_full_refresh = datetime.now()
            self.user_cache_invalidated.clear()
            logger.info(f"✅ Full refresh complete at {self.last_full_refresh}")

    def should_auto_refresh(self):
        """Check if auto-refresh is needed"""
        if not self.last_full_refresh:
            return True
        time_since_refresh = datetime.now() - self.last_full_refresh
        return time_since_refresh >= self.auto_refresh_interval

    def get_fresh_data(self, user_id: int = None, force: bool = False):
        """
        Get fresh data with smart caching
        - Auto-refreshes if interval passed
        - Force refreshes if requested
        - Per-user refresh if that user's cache is stale
        """
        # 1. Force refresh requested
        if force:
            logger.info(f"🔄 Force refresh requested (user: {user_id})")
            self._full_refresh()
            return self.database, self.cf_engine

        # 2. Check if user-specific refresh needed
        if user_id and user_id in self.user_cache_invalidated:
            logger.info(f"🔄 User {user_id} cache invalidated, refreshing...")
            self._full_refresh()
            return self.database, self.cf_engine

        # 3. Auto-refresh if interval passed
        if self.should_auto_refresh():
            logger.info(f"🔄 Auto-refresh triggered (interval: {self.auto_refresh_interval})")
            self._full_refresh()
            return self.database, self.cf_engine

        # 4. Use cached data
        return self.database, self.cf_engine

    def invalidate_user_cache(self, user_id: int):
        """Mark a specific user's cache as stale"""
        with self.lock:
            self.user_cache_invalidated.add(user_id)
            logger.info(f"📌 User {user_id} cache invalidated")

    def get_status(self):
        """Get refresh status"""
        if not self.last_full_refresh:
            return {"status": "initializing"}

        time_since = datetime.now() - self.last_full_refresh
        next_refresh = self.auto_refresh_interval - time_since

        return {
            "last_refresh": self.last_full_refresh.isoformat(),
            "seconds_since_refresh": int(time_since.total_seconds()),
            "next_auto_refresh_in_seconds": max(0, int(next_refresh.total_seconds())),
            "auto_refresh_interval_minutes": self.auto_refresh_interval.total_seconds() / 60,
            "invalidated_users": len(self.user_cache_invalidated)
        }

# ============================================================================
# Initialize FastAPI with Smart Data Manager
# ============================================================================
app = FastAPI(
    title="Boarding House Recommendation System - SMART REFRESH",
    description="Auto-refreshing Hybrid CF with Intelligent Caching",
    version="3.0.0-SMART"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Smart Data Manager (auto-refreshes every 5 minutes)
data_manager = SmartDataManager(auto_refresh_minutes=5)

logger.info("✅ Smart Data Manager initialized!")
logger.info("🔄 Auto-refresh: Every 5 minutes")
logger.info("⚡ Per-user cache invalidation: Enabled")


# ============================================================================
# Helper Functions (same as before)
# ============================================================================
def get_user_activity_profile(user_id: int, database) -> Dict:
    """Get user activity profile"""
    try:
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

        rating_values = [r['rating'] for r in ratings]
        avg_rating = sum(rating_values) / len(rating_values)
        variance = sum((r - avg_rating) ** 2 for r in rating_values) / len(rating_values)

        preference = database.get_user_preference(user_id)
        completeness = 0.0
        if preference:
            has_budget = 1 if preference.get('budget_min', 0) > 0 and preference.get('budget_max', 0) > 0 else 0
            has_distance = 1 if preference.get('preferred_distance', 0) > 0 else 0
            has_room_type = 1 if preference.get('room_type') and preference.get('room_type') != 'Any' else 0
            completeness = (has_budget + has_distance + has_room_type) / 3.0

        recent_rating_count = min(rating_count, 3)

        return {
            'rating_count': rating_count,
            'recent_rating_count': recent_rating_count,
            'recent_view_count': 0,
            'avg_rating': round(avg_rating, 2),
            'rating_variance': round(variance, 2),
            'preference_completeness': round(completeness, 2),
            'last_rating_time': ratings[-1].get('created_at') if ratings else None
        }
    except Exception as e:
        logger.error(f"Error getting user activity profile: {str(e)}")
        return {
            'rating_count': 0,
            'recent_rating_count': 0,
            'recent_view_count': 0,
            'avg_rating': 0,
            'rating_variance': 0,
            'preference_completeness': 0,
            'last_rating_time': None
        }


def determine_algorithm_strategy(user_id: int, database) -> Dict:
    """Determine algorithm strategy"""
    profile = get_user_activity_profile(user_id, database)

    rating_count = profile['rating_count']
    recent_ratings = profile['recent_rating_count']
    recent_views = profile['recent_view_count']
    variance = profile['rating_variance']
    completeness = profile['preference_completeness']

    if rating_count <= 1:
        return {
            'strategy': 'new_user',
            'algorithm': 'Popular',
            'reason': 'Showing popular properties based on student ratings',
            'action_needed': 'Rate 2+ properties to unlock personalized AI recommendations',
            'weights': {'user_based_cf': 0.0, 'item_based_cf': 0.0, 'content': 0.6, 'popular': 0.4}
        }

    if rating_count >= 2 and rating_count <= 4 and variance > 1.0:
        return {
            'strategy': 'exploring',
            'algorithm': 'Hybrid',
            'reason': 'Combining your preferences with similar property patterns',
            'action_needed': 'Continue rating properties to improve recommendations',
            'weights': {'user_based_cf': 0.1, 'item_based_cf': 0.5, 'content': 0.4, 'popular': 0.0}
        }

    if rating_count >= 5 and (recent_ratings > 0 or recent_views > 2):
        return {
            'strategy': 'active',
            'algorithm': 'Collaborative',
            'reason': 'AI analyzing students with similar preferences to yours',
            'action_needed': None,
            'weights': {'user_based_cf': 0.5, 'item_based_cf': 0.3, 'content': 0.2, 'popular': 0.0}
        }

    if rating_count >= 5 and recent_ratings == 0 and recent_views <= 2:
        return {
            'strategy': 'settled',
            'algorithm': 'Hybrid',
            'reason': 'Using balanced approach based on your established preferences',
            'action_needed': None,
            'weights': {'user_based_cf': 0.35, 'item_based_cf': 0.35, 'content': 0.3, 'popular': 0.0}
        }

    if completeness >= 0.67 and rating_count >= 2 and rating_count <= 6:
        return {
            'strategy': 'preference_focused',
            'algorithm': 'Content-Based',
            'reason': 'Matching properties to your detailed preference profile',
            'action_needed': None,
            'weights': {'user_based_cf': 0.2, 'item_based_cf': 0.2, 'content': 0.6, 'popular': 0.0}
        }

    return {
        'strategy': 'standard',
        'algorithm': 'Hybrid',
        'reason': 'Using multi-algorithm AI approach',
        'action_needed': None,
        'weights': {'user_based_cf': 0.4, 'item_based_cf': 0.3, 'content': 0.3, 'popular': 0.0}
    }


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
def read_root():
    status = data_manager.get_status()
    return {
        "status": "online",
        "mode": "SMART-REFRESH",
        "service": "Boarding House Recommendation System",
        "version": "3.0.0-SMART",
        "algorithm": "Adaptive Hybrid CF with Auto-Refresh",
        "refresh_status": status,
        "features": [
            "✅ Auto-refresh every 5 minutes",
            "✅ Per-user cache invalidation",
            "✅ Force refresh on demand",
            "✅ Thread-safe operations",
            "✅ No restart needed!"
        ]
    }


@app.get("/health")
def health_check():
    """Check service health"""
    try:
        database, _ = data_manager.get_fresh_data()
        database.test_connection()
        status = data_manager.get_status()

        return {
            "status": "healthy",
            "database": "connected",
            "service": "running",
            "refresh_status": status
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.get("/recommendations")
def get_recommendations(
    user_id: int,
    limit: int = 10,
    force_refresh: str = "false",
    include_rated: str = "false"
):
    """
    Get recommendations with smart auto-refresh
    - Automatically uses fresh data if cache expired
    - Force refresh if needed
    - Checks user-specific cache invalidation
    """
    try:
        force_refresh_bool = force_refresh.lower() in ['true', '1', 'yes']
        include_rated_bool = include_rated.lower() in ['true', '1', 'yes']

        # 🔥 Get fresh data (auto-refreshes if needed)
        database, cf_engine = data_manager.get_fresh_data(
            user_id=user_id,
            force=force_refresh_bool
        )

        strategy = determine_algorithm_strategy(user_id, database)
        strategy_name = strategy['strategy']
        adaptive_weights = strategy['weights']

        logger.info(f"User {user_id} - Strategy: {strategy_name}")

        # Generate recommendations
        if strategy_name == 'new_user':
            recommendations = cf_engine.get_cold_start_recommendations(
                user_id, limit, include_rated=include_rated_bool
            )
        elif strategy_name == 'active':
            recommendations = cf_engine.get_user_based_recommendations(
                user_id, limit, include_rated=include_rated_bool
            )
            if not recommendations:
                recommendations = cf_engine.get_hybrid_recommendations(
                    user_id, limit, adaptive_weights,
                    force_refresh=force_refresh_bool,
                    include_rated=include_rated_bool
                )
        else:
            recommendations = cf_engine.get_hybrid_recommendations(
                user_id, limit, adaptive_weights,
                force_refresh=force_refresh_bool,
                include_rated=include_rated_bool
            )

        if not recommendations:
            return {
                "recommendations": [],
                "algorithm_used": strategy['algorithm'],
                "strategy": strategy_name,
                "total_results": 0,
                "message": strategy['reason'],
                "action_needed": strategy.get('action_needed'),
                "user_activity": get_user_activity_profile(user_id, database),
                "refresh_status": data_manager.get_status()
            }

        return {
            "recommendations": recommendations,
            "algorithm_used": strategy['algorithm'],
            "strategy": strategy_name,
            "total_results": len(recommendations),
            "message": strategy['reason'],
            "action_needed": strategy.get('action_needed'),
            "user_activity": get_user_activity_profile(user_id, database),
            "weights_info": {
                "adaptive_weights": adaptive_weights,
                "content_breakdown": cf_engine.CONTENT_WEIGHTS
            },
            "refresh_status": data_manager.get_status()
        }

    except Exception as e:
        import traceback
        logger.error(f"Error generating recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/refresh")
def force_refresh():
    """Force immediate full refresh"""
    try:
        data_manager._full_refresh()
        return {
            "status": "success",
            "message": "Full data refresh completed",
            "refresh_status": data_manager.get_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-user-cache/{user_id}")
def clear_user_cache(user_id: int):
    """
    Invalidate cache for specific user
    Next request for this user will trigger refresh
    """
    try:
        data_manager.invalidate_user_cache(user_id)
        return {
            "status": "success",
            "message": f"User {user_id} cache invalidated",
            "user_id": user_id,
            "refresh_status": data_manager.get_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/refresh-status")
def get_refresh_status():
    """Get current refresh status"""
    return {
        "status": "success",
        "refresh_status": data_manager.get_status()
    }


@app.post("/retrain")
def retrain_model():
    """Alias for refresh (backward compatibility)"""
    return force_refresh()


# ============================================================================
# Additional endpoints (same as before, just use data_manager)
# ============================================================================
@app.get("/user-profile/{user_id}")
def get_user_profile(user_id: int):
    database, _ = data_manager.get_fresh_data(user_id=user_id)
    profile = get_user_activity_profile(user_id, database)
    strategy = determine_algorithm_strategy(user_id, database)
    return {"user_id": user_id, "profile": profile, "strategy": strategy}


@app.post("/predict-rating")
def predict_rating(request: dict):
    try:
        user_id = request.get('user_id')
        property_id = request.get('property_id')

        database, cf_engine = data_manager.get_fresh_data(user_id=user_id)
        prediction = cf_engine.predict_rating(user_id, property_id)
        strategy = determine_algorithm_strategy(user_id, database)

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


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  🚀 Starting Smart Auto-Refresh Recommendation Service")
    print("  Version 3.0.0-SMART")
    print("="*80)
    print(f"  🌐 Server: http://localhost:8001")
    print(f"  📚 Docs: http://localhost:8001/docs")
    print(f"  ❤️  Health: http://localhost:8001/health")
    print(f"\n  🔄 Smart Features:")
    print(f"     ✅ Auto-refresh: Every 5 minutes")
    print(f"     ✅ Per-user cache: Instant invalidation")
    print(f"     ✅ Force refresh: Available on demand")
    print(f"     ✅ Thread-safe: Concurrent request handling")
    print(f"\n  🎯 NO RESTART NEEDED - Data stays fresh automatically!")
    print("="*80 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
