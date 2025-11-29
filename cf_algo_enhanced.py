

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarUser:
    def __init__(self, user_id: int, similarity: float, common_ratings: int):
        self.user_id = user_id
        self.similarity = similarity
        self.common_ratings = common_ratings
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'similarity': round(self.similarity, 4),
            'common_ratings': self.common_ratings
        }


class SimilarProperty:
    def __init__(self, property_id: int, similarity: float, common_users: int):
        self.property_id = property_id
        self.similarity = similarity
        self.common_users = common_users
    
    def to_dict(self):
        return {
            'property_id': self.property_id,
            'similarity': round(self.similarity, 4),
            'common_users': self.common_users
        }


class EnhancedCollaborativeFilteringEngine:
    """
    Hybrid CF Engine with Research-Based Content Features
    
    Combines:
    1. User-Based CF (40%) - Similar students' preferences
    2. Item-Based CF (30%) - Similar properties
    3. Content-Based with Research Weights (30%) - Property features
    """
    
    # Research-based weights (total = 100%)
    WEIGHTS = {
        'distance': 0.30,      # Proximity to campus (30%)
        'cost': 0.25,          # Affordability (25%)
        'safety': 0.15,        # Safety & security (15%)
        'facilities': 0.10,    # Facilities & amenities (10%)
        'room_type': 0.10,     # Room type & privacy (10%)
        'management': 0.05,    # Management quality (5%)
        'social': 0.05         # Social environment (5%)
    }
    
    def __init__(self, database):
        self.db = database
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.cache = {}
        logger.info("Enhanced CF Engine initialized with research-based weights")
    
    def _build_user_item_matrix(self):
        """Build user-item rating matrix"""
        ratings_df = pd.DataFrame(self.db.get_all_ratings())
        
        if ratings_df.empty:
            return pd.DataFrame()
        
        matrix = ratings_df.pivot_table(
            index='user_id',
            columns='property_id',
            values='rating',
            fill_value=0
        )
        
        return matrix
    
    def _calculate_user_similarity(self):
        """Calculate user-user similarity using cosine similarity"""
        if self.user_item_matrix is None or self.user_item_matrix.empty:
            return None
        
        # Cosine similarity between users
        user_sim = cosine_similarity(self.user_item_matrix)
        user_sim_df = pd.DataFrame(
            user_sim,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        return user_sim_df
    
    def _calculate_item_similarity(self):
        """Calculate item-item similarity"""
        if self.user_item_matrix is None or self.user_item_matrix.empty:
            return None
        
        # Transpose and calculate similarity between properties
        item_sim = cosine_similarity(self.user_item_matrix.T)
        item_sim_df = pd.DataFrame(
            item_sim,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        return item_sim_df
    
    def _ensure_matrices(self):
        """Ensure similarity matrices are calculated"""
        if self.user_item_matrix is None:
            self.user_item_matrix = self._build_user_item_matrix()
        
        if self.user_similarity_matrix is None and not self.user_item_matrix.empty:
            self.user_similarity_matrix = self._calculate_user_similarity()
        
        if self.item_similarity_matrix is None and not self.user_item_matrix.empty:
            self.item_similarity_matrix = self._calculate_item_similarity()
    
    def _calculate_content_score(self, property_data: Dict, user_preference: Dict) -> float:
        """
        Calculate content-based score using research-based weights
        
        Returns normalized score 0-1
        """
        score = 0.0
        
        # 1. DISTANCE/PROXIMITY (30%)
        distance = property_data.get('distance_from_campus', 999)
        preferred_distance = user_preference.get('preferred_distance', 2.0)
        
        if distance <= preferred_distance:
            distance_score = 1.0
        elif distance <= preferred_distance * 1.5:
            distance_score = 0.6
        else:
            distance_score = max(0, 1.0 - (distance - preferred_distance) / 5.0)
        
        score += distance_score * self.WEIGHTS['distance']
        
        # 2. COST/AFFORDABILITY (25%)
        price = property_data.get('price', 0)
        budget_min = user_preference.get('budget_min', 0)
        budget_max = user_preference.get('budget_max', 999999)
        
        if budget_min <= price <= budget_max:
            cost_score = 1.0
        elif price < budget_min:
            cost_score = 0.7  # Below budget is still acceptable
        else:
            # Above budget - penalize
            cost_score = max(0, 1.0 - (price - budget_max) / budget_max)
        
        score += cost_score * self.WEIGHTS['cost']
        
        # 3. SAFETY & SECURITY (15%)
        # Proxy: average rating (higher ratings suggest safer/better managed)
        avg_rating = property_data.get('avg_rating', 3.0)
        safety_score = min(avg_rating / 5.0, 1.0)
        score += safety_score * self.WEIGHTS['safety']
        
        # 4. FACILITIES & AMENITIES (10%)
        user_amenities = set(user_preference.get('preferred_amenities', []))
        property_amenities = set(property_data.get('amenities', []))
        
        if user_amenities:
            amenity_match = len(user_amenities & property_amenities) / len(user_amenities)
        else:
            amenity_match = 0.5  # Neutral if no preference
        
        score += amenity_match * self.WEIGHTS['facilities']
        
        # 5. ROOM TYPE/PRIVACY (10%)
        user_room_type = user_preference.get('room_type', 'Any')
        property_room_type = property_data.get('room_type', 'Any')
        
        if user_room_type == 'Any' or property_room_type == 'Any':
            room_score = 0.8
        elif user_room_type == property_room_type:
            room_score = 1.0
        else:
            room_score = 0.4
        
        score += room_score * self.WEIGHTS['room_type']
        
        # 6. MANAGEMENT & MAINTENANCE (5%)
        # Proxy: rating count (more ratings = more established)
        rating_count = property_data.get('rating_count', 0)
        management_score = min(rating_count / 20.0, 1.0)
        score += management_score * self.WEIGHTS['management']
        
        # 7. SOCIAL & ENVIRONMENTAL (5%)
        user_gender_pref = user_preference.get('gender_preference', 'Any')
        property_gender = property_data.get('gender_restriction', 'Any')
        
        if user_gender_pref == 'Any' or property_gender == 'Any':
            social_score = 0.8
        elif user_gender_pref == property_gender:
            social_score = 1.0
        else:
            social_score = 0.3
        
        score += social_score * self.WEIGHTS['social']
        
        return score
    
    def get_hybrid_recommendations(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Get hybrid recommendations combining CF and content-based
        
        Weights:
        - User-Based CF: 40%
        - Item-Based CF: 30%
        - Content-Based: 30%
        """
        self._ensure_matrices()
        
        # Get user ratings
        user_ratings = self.db.get_user_ratings(user_id)
        rated_property_ids = [r['property_id'] for r in user_ratings]
        
        # Get all properties
        all_properties = self.db.get_all_properties()
        candidate_properties = [p for p in all_properties if p['id'] not in rated_property_ids]
        
        if not candidate_properties:
            return []
        
        # Get user preference
        user_preference = self.db.get_user_preference(user_id)
        if not user_preference:
            user_preference = {}
        
        recommendations = []
        
        for property_data in candidate_properties:
            property_id = property_data['id']
            
            # 1. USER-BASED CF SCORE (40%)
            user_cf_score = self._get_user_based_score(user_id, property_id)
            
            # 2. ITEM-BASED CF SCORE (30%)
            item_cf_score = self._get_item_based_score(user_id, property_id, user_ratings)
            
            # 3. CONTENT-BASED SCORE (30%)
            content_score = self._calculate_content_score(property_data, user_preference)
            
            # WEIGHTED COMBINATION
            final_score = (
                user_cf_score * 0.40 +
                item_cf_score * 0.30 +
                content_score * 0.30
            )
            
            # Convert to predicted rating (0-5 scale)
            predicted_rating = final_score * 5.0
            
            # Confidence based on CF availability
            confidence = self._calculate_confidence(user_cf_score, item_cf_score)
            
            recommendations.append({
                'property_id': property_id,
                'predicted_rating': round(predicted_rating, 2),
                'confidence': confidence,
                'algorithm': 'hybrid',
                'score_breakdown': {
                    'user_based': round(user_cf_score, 3),
                    'item_based': round(item_cf_score, 3),
                    'content': round(content_score, 3)
                }
            })
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        
        return recommendations[:limit]
    
    def _get_user_based_score(self, user_id: int, property_id: int) -> float:
        """Get user-based CF score (0-1)"""
        if self.user_similarity_matrix is None or user_id not in self.user_item_matrix.index:
            return 0.5  # Neutral score
        
        # Get similar users
        similar_users = self.find_similar_users(user_id, k=10)
        
        if not similar_users:
            return 0.5
        
        # Weighted average of similar users' ratings
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        for sim_user in similar_users:
            sim_user_id = sim_user.user_id
            similarity = sim_user.similarity
            
            # Check if similar user rated this property
            if property_id in self.user_item_matrix.columns:
                rating = self.user_item_matrix.loc[sim_user_id, property_id]
                if rating > 0:
                    weighted_sum += similarity * rating
                    similarity_sum += similarity
        
        if similarity_sum > 0:
            predicted_rating = weighted_sum / similarity_sum
            return min(predicted_rating / 5.0, 1.0)  # Normalize to 0-1
        
        return 0.5
    
    def _get_item_based_score(self, user_id: int, property_id: int, user_ratings: List[Dict]) -> float:
        """Get item-based CF score (0-1)"""
        if self.item_similarity_matrix is None or property_id not in self.item_similarity_matrix.index:
            return 0.5
        
        # Find properties user has rated highly (4+)
        highly_rated = [r for r in user_ratings if r['rating'] >= 4.0]
        
        if not highly_rated:
            return 0.5
        
        # Calculate similarity to highly rated properties
        similarities = []
        for rated_prop in highly_rated:
            rated_id = rated_prop['property_id']
            if rated_id in self.item_similarity_matrix.columns:
                sim = self.item_similarity_matrix.loc[property_id, rated_id]
                similarities.append(sim)
        
        if similarities:
            avg_similarity = np.mean(similarities)
            return max(0, min(avg_similarity, 1.0))
        
        return 0.5
    
    def _calculate_confidence(self, user_cf_score: float, item_cf_score: float) -> str:
        """Calculate confidence level"""
        # High confidence if both CF methods contributed
        if user_cf_score > 0.6 and item_cf_score > 0.6:
            return 'high'
        elif user_cf_score > 0.4 or item_cf_score > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def find_similar_users(self, user_id: int, k: int = 10) -> List[SimilarUser]:
        """Find k most similar users"""
        self._ensure_matrices()
        
        if self.user_similarity_matrix is None or user_id not in self.user_similarity_matrix.index:
            return []
        
        # Get similarities for this user
        similarities = self.user_similarity_matrix.loc[user_id].sort_values(ascending=False)
        
        # Exclude self and get top k
        similarities = similarities[similarities.index != user_id]
        top_similar = similarities.head(k)
        
        similar_users = []
        for sim_user_id, similarity in top_similar.items():
            if similarity > 0:
                # Count common ratings
                user_ratings = set(self.user_item_matrix.loc[user_id][self.user_item_matrix.loc[user_id] > 0].index)
                sim_ratings = set(self.user_item_matrix.loc[sim_user_id][self.user_item_matrix.loc[sim_user_id] > 0].index)
                common_count = len(user_ratings & sim_ratings)
                
                similar_users.append(SimilarUser(int(sim_user_id), float(similarity), common_count))
        
        return similar_users
    
    def find_similar_properties(self, property_id: int, k: int = 10) -> List[SimilarProperty]:
        """Find k most similar properties"""
        self._ensure_matrices()
        
        if self.item_similarity_matrix is None or property_id not in self.item_similarity_matrix.index:
            return []
        
        similarities = self.item_similarity_matrix.loc[property_id].sort_values(ascending=False)
        similarities = similarities[similarities.index != property_id]
        top_similar = similarities.head(k)
        
        similar_properties = []
        for sim_prop_id, similarity in top_similar.items():
            if similarity > 0:
                # Count common users
                prop_users = set(self.user_item_matrix[property_id][self.user_item_matrix[property_id] > 0].index)
                sim_users = set(self.user_item_matrix[sim_prop_id][self.user_item_matrix[sim_prop_id] > 0].index)
                common_count = len(prop_users & sim_users)
                
                similar_properties.append(SimilarProperty(int(sim_prop_id), float(similarity), common_count))
        
        return similar_properties
    
    def get_user_based_recommendations(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Pure user-based CF recommendations"""
        self._ensure_matrices()
        
        similar_users = self.find_similar_users(user_id, k=20)
        
        if not similar_users:
            return []
        
        user_ratings = self.db.get_user_ratings(user_id)
        rated_ids = [r['property_id'] for r in user_ratings]
        
        # Aggregate ratings from similar users
        property_scores = {}
        
        for sim_user in similar_users:
            sim_ratings = self.db.get_user_ratings(sim_user.user_id)
            for rating in sim_ratings:
                prop_id = rating['property_id']
                if prop_id not in rated_ids:
                    if prop_id not in property_scores:
                        property_scores[prop_id] = {'sum': 0, 'weight': 0}
                    property_scores[prop_id]['sum'] += rating['rating'] * sim_user.similarity
                    property_scores[prop_id]['weight'] += sim_user.similarity
        
        # Calculate predictions
        recommendations = []
        for prop_id, scores in property_scores.items():
            if scores['weight'] > 0:
                predicted_rating = scores['sum'] / scores['weight']
                recommendations.append({
                    'property_id': prop_id,
                    'predicted_rating': round(predicted_rating, 2),
                    'confidence': 'medium'
                })
        
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:limit]
    
    def get_item_based_recommendations(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Pure item-based CF recommendations"""
        self._ensure_matrices()
        
        user_ratings = self.db.get_user_ratings(user_id)
        if not user_ratings:
            return []
        
        rated_ids = [r['property_id'] for r in user_ratings]
        highly_rated = [r for r in user_ratings if r['rating'] >= 4.0]
        
        if not highly_rated:
            return []
        
        # Find similar properties
        property_scores = {}
        
        for rating in highly_rated:
            similar_props = self.find_similar_properties(rating['property_id'], k=20)
            for sim_prop in similar_props:
                if sim_prop.property_id not in rated_ids:
                    if sim_prop.property_id not in property_scores:
                        property_scores[sim_prop.property_id] = 0
                    property_scores[sim_prop.property_id] += sim_prop.similarity * rating['rating']
        
        recommendations = []
        for prop_id, score in property_scores.items():
            recommendations.append({
                'property_id': prop_id,
                'predicted_rating': round(min(score, 5.0), 2),
                'confidence': 'medium'
            })
        
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:limit]
    
    def predict_rating(self, user_id: int, property_id: int) -> float:
        """Predict rating for user-property pair"""
        # Use hybrid approach
        recommendations = self.get_hybrid_recommendations(user_id, limit=100)
        
        for rec in recommendations:
            if rec['property_id'] == property_id:
                return rec['predicted_rating']
        
        # Fallback to 3.0 (neutral)
        return 3.0
    
    def clear_cache(self):
        """Clear cached matrices"""
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.cache = {}
        logger.info("Cache cleared")
