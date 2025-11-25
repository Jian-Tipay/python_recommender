import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math

import models


class EnhancedCollaborativeFilteringEngine:
    """
    Enhanced CF Engine that combines:
    1. User-Based CF (similar users)
    2. Item-Based CF (similar properties) 
    3. Content Features (amenities of highly-rated properties)
    """
    
    def __init__(self, db, min_common_ratings=2, k_neighbors=10):
        self.db = db
        self.min_common_ratings = min_common_ratings
        self.k_neighbors = k_neighbors
        self.rating_matrix = None
        self.user_means = {}
        self.global_mean = 3.5
        
    def _build_rating_matrix(self):
        """Build user-item rating matrix from database"""
        ratings_data = self.db.get_all_ratings()
        
        if not ratings_data:
            return models.RatingMatrix(users=[], properties=[], ratings={})
        
        users = set()
        properties = set()
        ratings = {}
        rating_sum = 0
        rating_count = 0
        
        for r in ratings_data:
            user_id = r['user_id']
            property_id = r['property_id']
            rating = float(r['rating'])
            
            users.add(user_id)
            properties.add(property_id)
            ratings[(user_id, property_id)] = rating
            rating_sum += rating
            rating_count += 1
        
        self.global_mean = rating_sum / rating_count if rating_count > 0 else 3.5
        
        for user_id in users:
            user_ratings = [r for (u, p), r in ratings.items() if u == user_id]
            self.user_means[user_id] = np.mean(user_ratings) if user_ratings else self.global_mean
        
        return models.RatingMatrix(
            users=list(users),
            properties=list(properties),
            ratings=ratings
        )
    
    def _get_rating_matrix(self) -> models.RatingMatrix:
        """Get or build rating matrix (with caching)"""
        if self.rating_matrix is None:
            self.rating_matrix = self._build_rating_matrix()
        return self.rating_matrix
    
    def _pearson_correlation(self, ratings1: dict, ratings2: dict) -> float:
        """Calculate Pearson correlation coefficient"""
        common_items = set(ratings1.keys()) & set(ratings2.keys())
        
        if len(common_items) < self.min_common_ratings:
            return 0.0
        
        ratings1_common = [ratings1[item] for item in common_items]
        ratings2_common = [ratings2[item] for item in common_items]
        
        mean1 = np.mean(ratings1_common)
        mean2 = np.mean(ratings2_common)
        
        numerator = sum((r1 - mean1) * (r2 - mean2) 
                       for r1, r2 in zip(ratings1_common, ratings2_common))
        
        denominator1 = math.sqrt(sum((r1 - mean1) ** 2 for r1 in ratings1_common))
        denominator2 = math.sqrt(sum((r2 - mean2) ** 2 for r2 in ratings2_common))
        
        if denominator1 == 0 or denominator2 == 0:
            return 0.0
        
        correlation = numerator / (denominator1 * denominator2)
        return max(-1.0, min(1.0, correlation))
    
    def _cosine_similarity(self, ratings1: dict, ratings2: dict) -> float:
        """Calculate cosine similarity"""
        common_items = set(ratings1.keys()) & set(ratings2.keys())
        
        if len(common_items) < self.min_common_ratings:
            return 0.0
        
        vec1 = [ratings1[item] for item in common_items]
        vec2 = [ratings2[item] for item in common_items]
        
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(v ** 2 for v in vec1))
        magnitude2 = math.sqrt(sum(v ** 2 for v in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def find_similar_users(self, user_id: int, k: int = None) -> List[models.SimilarUser]:
        """Find k most similar users based on rating patterns"""
        if k is None:
            k = self.k_neighbors
        
        matrix = self._get_rating_matrix()
        
        if not matrix.has_user(user_id):
            return []
        
        user_ratings = matrix.get_user_ratings(user_id)
        
        if not user_ratings:
            return []
        
        similarities = []
        
        for other_user_id in matrix.users:
            if other_user_id == user_id:
                continue
            
            other_ratings = matrix.get_user_ratings(other_user_id)
            
            if not other_ratings:
                continue
            
            similarity = self._pearson_correlation(user_ratings, other_ratings)
            
            if similarity > 0:
                common_items = set(user_ratings.keys()) & set(other_ratings.keys())
                similarities.append(models.SimilarUser(
                    user_id=other_user_id,
                    similarity=similarity,
                    common_ratings=len(common_items)
                ))
        
        similarities.sort(key=lambda x: x.similarity, reverse=True)
        return similarities[:k]
    
    def find_similar_properties(self, property_id: int, k: int = None) -> List[models.SimilarProperty]:
        """Find k most similar properties based on user ratings"""
        if k is None:
            k = self.k_neighbors
        
        matrix = self._get_rating_matrix()
        
        if not matrix.has_property(property_id):
            return []
        
        property_ratings = matrix.get_property_ratings(property_id)
        
        if not property_ratings:
            return []
        
        similarities = []
        
        for other_property_id in matrix.properties:
            if other_property_id == property_id:
                continue
            
            other_ratings = matrix.get_property_ratings(other_property_id)
            
            if not other_ratings:
                continue
            
            similarity = self._cosine_similarity(property_ratings, other_ratings)
            
            if similarity > 0:
                common_users = set(property_ratings.keys()) & set(other_ratings.keys())
                similarities.append(models.SimilarProperty(
                    property_id=other_property_id,
                    similarity=similarity,
                    common_users=len(common_users)
                ))
        
        similarities.sort(key=lambda x: x.similarity, reverse=True)
        return similarities[:k]
    
    def _get_user_liked_amenities(self, user_id: int, min_rating: float = 3.5) -> Dict[int, float]:
        """
        Get amenities from properties user rated highly (3.5+ stars)
        Returns: {amenity_id: importance_score}
        """
        matrix = self._get_rating_matrix()
        user_ratings = matrix.get_user_ratings(user_id)
        
        amenity_scores = defaultdict(float)
        total_weight = 0
        
        for property_id, rating in user_ratings.items():
            if rating >= min_rating:
                # Get amenities for this property
                amenities = self.db.get_property_amenities(property_id)
                
                # Weight by rating (5-star property amenities are more important)
                weight = rating / 5.0
                total_weight += weight
                
                for amenity in amenities:
                    amenity_scores[amenity['amenity_id']] += weight
        
        # Normalize scores
        if total_weight > 0:
            for amenity_id in amenity_scores:
                amenity_scores[amenity_id] /= total_weight
        
        return dict(amenity_scores)
    
    def _calculate_content_similarity(self, property_id: int, liked_amenities: Dict[int, float]) -> float:
        """
        Calculate how well a property matches user's liked amenities
        Returns score 0-1
        """
        if not liked_amenities:
            return 0.0
        
        property_amenities = self.db.get_property_amenities(property_id)
        property_amenity_ids = set(a['amenity_id'] for a in property_amenities)
        
        if not property_amenity_ids:
            return 0.0
        
        # Calculate match score
        match_score = 0
        for amenity_id in property_amenity_ids:
            if amenity_id in liked_amenities:
                match_score += liked_amenities[amenity_id]
        
        # Normalize by number of property amenities
        return min(1.0, match_score)
    
    def get_hybrid_recommendations(self, user_id: int, limit: int = 10) -> List[dict]:
        """
        Enhanced hybrid recommendations combining:
        1. User-Based CF (40%)
        2. Item-Based CF (30%)  
        3. Content Features from highly-rated properties (30%)
        """
        matrix = self._get_rating_matrix()
        
        if not matrix.has_user(user_id):
            return []
        
        # Get properties already rated
        rated_properties = set(self.db.get_user_rated_properties(user_id))
        all_properties = self.db.get_active_properties(exclude_ids=list(rated_properties))
        
        if not all_properties:
            return []
        
        # Get user's liked amenities from highly-rated properties
        liked_amenities = self._get_user_liked_amenities(user_id, min_rating=3.5)
        
        # Find similar users
        similar_users = self.find_similar_users(user_id, k=self.k_neighbors)
        
        # Get user's ratings for item-based CF
        user_ratings = matrix.get_user_ratings(user_id)
        
        recommendations = {}
        
        for property_data in all_properties:
            property_id = property_data['id']
            scores = {
                'user_based': 0,
                'item_based': 0,
                'content': 0
            }
            
            # 1. User-Based CF Score (40%)
            if similar_users:
                weighted_sum = 0
                similarity_sum = 0
                
                for similar_user in similar_users:
                    rating = matrix.get_rating(similar_user.user_id, property_id)
                    if rating is not None:
                        weighted_sum += similar_user.similarity * rating
                        similarity_sum += abs(similar_user.similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    scores['user_based'] = (predicted_rating / 5.0) * 0.4
            
            # 2. Item-Based CF Score (30%)
            similar_properties = self.find_similar_properties(property_id, k=self.k_neighbors)
            
            if similar_properties and user_ratings:
                weighted_sum = 0
                similarity_sum = 0
                
                for similar_prop in similar_properties:
                    if similar_prop.property_id in user_ratings:
                        user_rating = user_ratings[similar_prop.property_id]
                        weighted_sum += similar_prop.similarity * user_rating
                        similarity_sum += abs(similar_prop.similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    scores['item_based'] = (predicted_rating / 5.0) * 0.3
            
            # 3. Content Feature Score (30%) - NEW!
            if liked_amenities:
                content_score = self._calculate_content_similarity(property_id, liked_amenities)
                scores['content'] = content_score * 0.3
            
            # Calculate final score
            final_score = sum(scores.values())
            
            if final_score > 0:
                recommendations[property_id] = {
                    'property_id': property_id,
                    'title': property_data['title'],
                    'address': property_data['address'],
                    'price': float(property_data['price']),
                    'distance_from_campus': float(property_data['distance_from_campus']) if property_data.get('distance_from_campus') else None,
                    'predicted_rating': round(final_score * 5, 2),  # Scale back to 1-5
                    'confidence': round(min(1.0, sum(1 for s in scores.values() if s > 0) / 3), 2),
                    'algorithm': 'enhanced_hybrid_cf',
                    'score_breakdown': {
                        'user_based': round(scores['user_based'], 3),
                        'item_based': round(scores['item_based'], 3),
                        'content': round(scores['content'], 3)
                    }
                }
        
        # Sort by final score
        sorted_recs = sorted(recommendations.values(), 
                           key=lambda x: x['predicted_rating'], 
                           reverse=True)
        
        return sorted_recs[:limit]
    
    def get_user_based_recommendations(self, user_id: int, limit: int = 10) -> List[dict]:
        """Pure User-Based CF"""
        matrix = self._get_rating_matrix()
        
        if not matrix.has_user(user_id):
            return []
        
        rated_properties = set(self.db.get_user_rated_properties(user_id))
        all_properties = self.db.get_active_properties(exclude_ids=list(rated_properties))
        
        if not all_properties:
            return []
        
        similar_users = self.find_similar_users(user_id, k=self.k_neighbors)
        
        if not similar_users:
            return []
        
        predictions = []
        
        for property_data in all_properties:
            property_id = property_data['id']
            
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_user in similar_users:
                rating = matrix.get_rating(similar_user.user_id, property_id)
                if rating is not None:
                    weighted_sum += similar_user.similarity * rating
                    similarity_sum += abs(similar_user.similarity)
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                confidence = min(1.0, similarity_sum / len(similar_users))
                
                predictions.append({
                    'property_id': property_id,
                    'title': property_data['title'],
                    'address': property_data['address'],
                    'price': float(property_data['price']),
                    'distance_from_campus': float(property_data['distance_from_campus']) if property_data.get('distance_from_campus') else None,
                    'predicted_rating': round(predicted_rating, 2),
                    'confidence': round(confidence, 2),
                    'algorithm': 'user_based_cf'
                })
        
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:limit]
    
    def get_item_based_recommendations(self, user_id: int, limit: int = 10) -> List[dict]:
        """Pure Item-Based CF"""
        matrix = self._get_rating_matrix()
        
        if not matrix.has_user(user_id):
            return []
        
        user_ratings = matrix.get_user_ratings(user_id)
        
        if not user_ratings:
            return []
        
        rated_properties = set(user_ratings.keys())
        all_properties = self.db.get_active_properties(exclude_ids=list(rated_properties))
        
        if not all_properties:
            return []
        
        predictions = []
        
        for property_data in all_properties:
            property_id = property_data['id']
            
            similar_properties = self.find_similar_properties(property_id, k=self.k_neighbors)
            
            if not similar_properties:
                continue
            
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_prop in similar_properties:
                if similar_prop.property_id in user_ratings:
                    user_rating = user_ratings[similar_prop.property_id]
                    weighted_sum += similar_prop.similarity * user_rating
                    similarity_sum += abs(similar_prop.similarity)
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                confidence = min(1.0, similarity_sum / len(similar_properties))
                
                predictions.append({
                    'property_id': property_id,
                    'title': property_data['title'],
                    'address': property_data['address'],
                    'price': float(property_data['price']),
                    'distance_from_campus': float(property_data['distance_from_campus']) if property_data.get('distance_from_campus') else None,
                    'predicted_rating': round(predicted_rating, 2),
                    'confidence': round(confidence, 2),
                    'algorithm': 'item_based_cf'
                })
        
        predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return predictions[:limit]
    
    def predict_rating(self, user_id: int, property_id: int) -> float:
        """Predict rating for a user-property pair"""
        matrix = self._get_rating_matrix()
        
        if not matrix.has_user(user_id) or not matrix.has_property(property_id):
            return self.global_mean
        
        existing_rating = matrix.get_rating(user_id, property_id)
        if existing_rating is not None:
            return existing_rating
        
        similar_users = self.find_similar_users(user_id, k=self.k_neighbors)
        
        if not similar_users:
            return self.user_means.get(user_id, self.global_mean)
        
        weighted_sum = 0
        similarity_sum = 0
        
        for similar_user in similar_users:
            rating = matrix.get_rating(similar_user.user_id, property_id)
            if rating is not None:
                weighted_sum += similar_user.similarity * rating
                similarity_sum += abs(similar_user.similarity)
        
        if similarity_sum == 0:
            return self.user_means.get(user_id, self.global_mean)
        
        predicted_rating = weighted_sum / similarity_sum
        return max(1.0, min(5.0, predicted_rating))
    
    def clear_cache(self):
        """Clear cached rating matrix"""
        self.rating_matrix = None
        self.user_means = {}
        self.global_mean = 3.5