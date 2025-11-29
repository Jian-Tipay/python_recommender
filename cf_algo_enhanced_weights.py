import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
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
    Adaptive Hybrid CF Engine with Research-Based Content Features
    
    Supports dynamic weight adjustment based on user behavior:
    - NEW USER → Popular (60% content, 40% popular)
    - EXPLORING → Item+Content Hybrid (50% item, 40% content, 10% user)
    - ACTIVE → Collaborative (50% user, 30% item, 20% content)
    - SETTLED → Balanced Hybrid (35% user, 35% item, 30% content)
    - PREFERENCE-FOCUSED → Content-Based (60% content, 20% user, 20% item)
    
    All scores are on 0-100 scale to match database storage
    """
    
    # Research-based content weights (total = 100%)
    CONTENT_WEIGHTS = {
        'distance': 30.0,      # Proximity to campus (30 points)
        'cost': 25.0,          # Affordability (25 points)
        'safety': 15.0,        # Safety & security (15 points)
        'facilities': 10.0,    # Facilities & amenities (10 points)
        'room_type': 10.0,     # Room type & privacy (10 points)
        'management': 5.0,     # Management quality (5 points)
        'social': 5.0          # Social environment (5 points)
    }
    
    # Default algorithm weights (can be overridden)
    DEFAULT_WEIGHTS = {
        'user_based_cf': 0.40,
        'item_based_cf': 0.30,
        'content': 0.30,
        'popular': 0.00
    }
    
    def __init__(self, database):
        self.db = database
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.cache = {}
        logger.info("Enhanced CF Engine initialized with adaptive weight support")
    
    def _build_user_item_matrix(self):
        """Build user-item rating matrix"""
        try:
            ratings_df = pd.DataFrame(self.db.get_all_ratings())
            
            if ratings_df.empty:
                logger.warning("No ratings found in database")
                return pd.DataFrame()
            
            # Ensure proper data types
            ratings_df['user_id'] = ratings_df['user_id'].astype(int)
            ratings_df['property_id'] = ratings_df['property_id'].astype(int)
            ratings_df['rating'] = ratings_df['rating'].astype(float)
            
            matrix = ratings_df.pivot_table(
                index='user_id',
                columns='property_id',
                values='rating',
                fill_value=0.0
            )
            
            logger.info(f"Built user-item matrix: {matrix.shape[0]} users x {matrix.shape[1]} properties")
            return matrix
        except Exception as e:
            logger.error(f"Error building user-item matrix: {str(e)}")
            return pd.DataFrame()
    
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
        
        Returns score on 0-100 scale
        """
        try:
            score = 0.0
            
            # 1. DISTANCE/PROXIMITY (30 points)
            distance = float(property_data.get('distance_from_campus', 999))
            preferred_distance = float(user_preference.get('preferred_distance', 2.0))
            
            if distance <= preferred_distance:
                distance_score = 1.0
            elif distance <= preferred_distance * 1.5:
                distance_score = 0.6
            else:
                distance_score = max(0, 1.0 - (distance - preferred_distance) / 5.0)
            
            score += distance_score * self.CONTENT_WEIGHTS['distance']
            
            # 2. COST/AFFORDABILITY (25 points)
            price = float(property_data.get('price', 0))
            budget_min = float(user_preference.get('budget_min', 0))
            budget_max = float(user_preference.get('budget_max', 999999))
            
            if budget_min <= price <= budget_max:
                cost_score = 1.0
            elif price < budget_min:
                cost_score = 0.7  # Below budget is still acceptable
            else:
                # Above budget - penalize
                if budget_max > 0:
                    cost_score = max(0, 1.0 - (price - budget_max) / budget_max)
                else:
                    cost_score = 0.3
            
            score += cost_score * self.CONTENT_WEIGHTS['cost']
            
            # 3. SAFETY & SECURITY (15 points)
            # Proxy: average rating (higher ratings suggest safer/better managed)
            avg_rating = float(property_data.get('avg_rating', 3.0))
            safety_score = min(avg_rating / 5.0, 1.0)
            score += safety_score * self.CONTENT_WEIGHTS['safety']
            
            # 4. FACILITIES & AMENITIES (10 points)
            user_amenities = set(user_preference.get('preferred_amenities', []))
            property_amenities = set(property_data.get('amenities', []))
            
            if user_amenities:
                amenity_match = len(user_amenities & property_amenities) / len(user_amenities)
            else:
                amenity_match = 0.5  # Neutral if no preference
            
            score += amenity_match * self.CONTENT_WEIGHTS['facilities']
            
            # 5. ROOM TYPE/PRIVACY (10 points)
            user_room_type = str(user_preference.get('room_type', 'Any'))
            property_room_type = str(property_data.get('room_type', 'Any'))
            
            if user_room_type == 'Any' or property_room_type == 'Any':
                room_score = 0.8
            elif user_room_type == property_room_type:
                room_score = 1.0
            else:
                room_score = 0.4
            
            score += room_score * self.CONTENT_WEIGHTS['room_type']
            
            # 6. MANAGEMENT & MAINTENANCE (5 points)
            # Proxy: rating count (more ratings = more established)
            rating_count = int(property_data.get('rating_count', 0))
            management_score = min(rating_count / 20.0, 1.0)
            score += management_score * self.CONTENT_WEIGHTS['management']
            
            # 7. SOCIAL & ENVIRONMENTAL (5 points)
            user_gender_pref = str(user_preference.get('gender_preference', 'Any'))
            property_gender = str(property_data.get('gender_preference', 'Any'))
            
            if user_gender_pref == 'Any' or property_gender == 'Any':
                social_score = 0.8
            elif user_gender_pref == property_gender:
                social_score = 1.0
            else:
                social_score = 0.3
            
            score += social_score * self.CONTENT_WEIGHTS['social']
            
            # Ensure score is within 0-100 range
            return max(0.0, min(100.0, round(score, 2)))
            
        except Exception as e:
            logger.error(f"Error calculating content score: {str(e)}")
            return 50.0  # Return neutral score on error
    
    def _generate_explanation(self, breakdown: Dict, detected_algorithm: str) -> str:
        """Generate human-readable explanation"""
        explanations = []
        
        user_score = breakdown.get('user_based', 0)
        item_score = breakdown.get('item_based', 0)
        content_score = breakdown.get('content', 0)
        
        if detected_algorithm == 'Collaborative':
            explanations.append('Highly rated by students with similar preferences to yours')
        elif detected_algorithm == 'Hybrid':
            if user_score > 20:
                explanations.append('Students like you rated this highly')
            if item_score > 15:
                explanations.append('Similar to properties you liked')
            if content_score > 15:
                explanations.append('Matches your preferences')
        elif detected_algorithm == 'Content-Based':
            explanations.append('Strongly matches your set preferences')
        elif detected_algorithm == 'Popular':
            explanations.append('Highly rated by many students')
        
        return '. '.join(explanations) if explanations else 'Recommended for you'
    
    def _detect_algorithm_from_scores(self, breakdown: Dict) -> str:
        """
        Detect algorithm type based on score breakdown
        Matches Laravel's detectAlgorithmType logic exactly
        """
        user_score = breakdown.get('user_based', 0)
        item_score = breakdown.get('item_based', 0)
        content_score = breakdown.get('content', 0)
        
        user_significant = user_score > 20
        item_significant = item_score > 15
        content_significant = content_score > 15
        
        if user_significant and not item_significant and not content_significant:
            return 'Collaborative'
        elif user_significant and (item_significant or content_significant):
            return 'Hybrid'
        elif item_significant and content_significant:
            return 'Hybrid'
        elif content_significant and not user_significant and not item_significant:
            return 'Content-Based'
        else:
            return 'Hybrid'
    
    def get_hybrid_recommendations(
        self, 
        user_id: int, 
        limit: int = 10, 
        adaptive_weights: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get hybrid recommendations with ADAPTIVE WEIGHTS
        
        Args:
            user_id: User ID
            limit: Number of recommendations
            adaptive_weights: Dict with keys: user_based_cf, item_based_cf, content, popular
                             If None, uses DEFAULT_WEIGHTS (40-30-30)
        
        Returns scores on 0-100 scale with per-property algorithm detection
        """
        try:
            self._ensure_matrices()
            
            # Use adaptive weights if provided, otherwise use defaults
            if adaptive_weights is None:
                adaptive_weights = self.DEFAULT_WEIGHTS.copy()
            
            # Convert to percentage points (0-100 scale)
            user_weight = adaptive_weights.get('user_based_cf', 0.4) * 100
            item_weight = adaptive_weights.get('item_based_cf', 0.3) * 100
            content_weight = adaptive_weights.get('content', 0.3) * 100
            
            logger.info(f"Using adaptive weights: User={user_weight}%, Item={item_weight}%, Content={content_weight}%")
            
            # Get user ratings
            user_ratings = self.db.get_user_ratings(user_id)
            rated_property_ids = [r['property_id'] for r in user_ratings]
            
            # Get all properties
            all_properties = self.db.get_all_properties()
            candidate_properties = [p for p in all_properties if p['id'] not in rated_property_ids]
            
            if not candidate_properties:
                logger.info(f"No candidate properties for user {user_id}")
                return []
            
            # Get user preference
            user_preference = self.db.get_user_preference(user_id)
            if not user_preference:
                user_preference = {}
                logger.info(f"No user preference found for user {user_id}, using defaults")
            
            recommendations = []
            
            for property_data in candidate_properties:
                try:
                    property_id = property_data['id']
                    
                    # 1. USER-BASED CF SCORE
                    user_cf_score_normalized = self._get_user_based_score(user_id, property_id)
                    user_cf_score = user_cf_score_normalized * user_weight
                    
                    # 2. ITEM-BASED CF SCORE
                    item_cf_score_normalized = self._get_item_based_score(user_id, property_id, user_ratings)
                    item_cf_score = item_cf_score_normalized * item_weight
                    
                    # 3. CONTENT-BASED SCORE
                    content_score_full = self._calculate_content_score(property_data, user_preference)
                    content_score = (content_score_full / 100.0) * content_weight
                    
                    # WEIGHTED COMBINATION (0-100 scale)
                    final_score = user_cf_score + item_cf_score + content_score
                    
                    # Convert to predicted rating (0-5 scale)
                    predicted_rating = (final_score / 100.0) * 5.0
                    
                    # Score breakdown for algorithm detection
                    breakdown = {
                        'user_based': round(user_cf_score, 2),
                        'item_based': round(item_cf_score, 2),
                        'content': round(content_score, 2)
                    }
                    
                    # Detect algorithm from actual scores
                    detected_algorithm = self._detect_algorithm_from_scores(breakdown)
                    
                    # Generate explanation
                    explanation = self._generate_explanation(breakdown, detected_algorithm)
                    
                    # Confidence based on CF availability
                    confidence = self._calculate_confidence(user_cf_score_normalized, item_cf_score_normalized)
                    
                    recommendations.append({
                        'property_id': property_id,
                        'predicted_rating': round(predicted_rating, 2),
                        'confidence': confidence,
                        'algorithm': detected_algorithm,
                        'explanation': explanation,
                        'score_breakdown': breakdown,
                        'total_score': round(final_score, 2)
                    })
                except Exception as e:
                    logger.error(f"Error processing property {property_data.get('id', 'unknown')}: {str(e)}")
                    continue
            
            # Sort by total score
            recommendations.sort(key=lambda x: x['total_score'], reverse=True)
            
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in get_hybrid_recommendations for user {user_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def get_cold_start_recommendations(self, user_id: int, limit: int = 10) -> List[Dict]:
        """
        Cold start recommendations for new users (0-1 ratings)
        Uses popularity + basic content matching
        """
        try:
            user_preference = self.db.get_user_preference(user_id)
            
            all_properties = self.db.get_all_properties()
            
            recommendations = []
            
            for property_data in all_properties:
                # Calculate popularity score
                avg_rating = float(property_data.get('avg_rating', 0))
                rating_count = int(property_data.get('rating_count', 0))
                
                # Popularity = (avg_rating * 12) + (rating_count * 2), max 100
                popularity_score = min((avg_rating * 12) + (rating_count * 2), 100)
                
                # Basic content score if preferences exist
                content_score = 50.0  # Default neutral
                if user_preference:
                    content_score = self._calculate_content_score(property_data, user_preference)
                
                # Weighted combination: 40% popular, 60% content
                final_score = (popularity_score * 0.4) + (content_score * 0.6)
                predicted_rating = (final_score / 100.0) * 5.0
                
                recommendations.append({
                    'property_id': property_data['id'],
                    'predicted_rating': round(predicted_rating, 2),
                    'confidence': 'low',
                    'algorithm': 'Popular',
                    'explanation': 'Highly rated by many students',
                    'score_breakdown': {
                        'user_based': 0.0,
                        'item_based': 0.0,
                        'content': round(content_score * 0.6, 2)
                    },
                    'total_score': round(final_score, 2)
                })
            
            recommendations.sort(key=lambda x: x['total_score'], reverse=True)
            logger.info(f"Generated {len(recommendations)} cold start recommendations for user {user_id}")
            return recommendations[:limit]
            
        except Exception as e:
            logger.error(f"Error in get_cold_start_recommendations: {str(e)}")
            return []
    
    def _get_user_based_score(self, user_id: int, property_id: int) -> float:
        """Get user-based CF score (0-1 normalized)"""
        try:
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
        except Exception as e:
            logger.error(f"Error in _get_user_based_score: {str(e)}")
            return 0.5
    
    def _get_item_based_score(self, user_id: int, property_id: int, user_ratings: List[Dict]) -> float:
        """Get item-based CF score (0-1 normalized)"""
        try:
            if self.item_similarity_matrix is None or property_id not in self.item_similarity_matrix.index:
                return 0.5
            
            # Find properties user has rated highly (4+)
            highly_rated = [r for r in user_ratings if float(r.get('rating', 0)) >= 4.0]
            
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
        except Exception as e:
            logger.error(f"Error in _get_item_based_score: {str(e)}")
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
                final_score = (predicted_rating / 5.0) * 100
                
                recommendations.append({
                    'property_id': prop_id,
                    'predicted_rating': round(predicted_rating, 2),
                    'confidence': 'medium',
                    'algorithm': 'Collaborative',
                    'explanation': 'Highly rated by students with similar preferences to yours',
                    'score_breakdown': {
                        'user_based': round(final_score * 0.5, 2),
                        'item_based': round(final_score * 0.3, 2),
                        'content': round(final_score * 0.2, 2)
                    },
                    'total_score': round(final_score, 2)
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
            predicted_rating = min(score, 5.0)
            final_score = (predicted_rating / 5.0) * 100
            
            recommendations.append({
                'property_id': prop_id,
                'predicted_rating': round(predicted_rating, 2),
                'confidence': 'medium',
                'algorithm': 'Hybrid',
                'explanation': 'Similar to properties you liked',
                'score_breakdown': {
                    'user_based': round(final_score * 0.1, 2),
                    'item_based': round(final_score * 0.5, 2),
                    'content': round(final_score * 0.4, 2)
                },
                'total_score': round(final_score, 2)
            })
        
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:limit]
    
    def predict_rating(self, user_id: int, property_id: int) -> float:
        """Predict rating for user-property pair"""
        # Use hybrid approach with default weights
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
