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

    FIXED: Now properly refreshes user preferences and handles gender matching
    """

    CONTENT_WEIGHTS = {
        'distance': 30.0,
        'cost': 25.0,
        'safety': 15.0,
        'facilities': 10.0,
        'room_type': 10.0,
        'management': 5.0,
        'social': 5.0
    }

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
        self.preference_cache = {}
        logger.info("Enhanced CF Engine initialized with preference caching")

    def _get_fresh_user_preference(self, user_id: int, force_refresh: bool = False) -> Dict:
        """Get user preference with caching and force refresh option"""
        cache_key = f"pref_{user_id}"

        try:
            preference = self.db.get_user_preference(user_id)
            if preference:
                self.preference_cache[cache_key] = preference
                logger.info(f"Loaded fresh preferences for user {user_id}")
            return preference or {}
        except Exception as e:
            logger.error(f"Error loading preferences for user {user_id}: {str(e)}")
            return self.preference_cache.get(cache_key, {})

    def _build_user_item_matrix(self):
        """Build user-item rating matrix"""
        try:
            ratings_df = pd.DataFrame(self.db.get_all_ratings())

            if ratings_df.empty:
                logger.warning("No ratings found in database")
                return pd.DataFrame()

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

        item_sim = cosine_similarity(self.user_item_matrix.T)
        item_sim_df = pd.DataFrame(
            item_sim,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

        return item_sim_df

    def _ensure_matrices(self, force_rebuild: bool = False):
        """Ensure similarity matrices are calculated"""
        if self.user_item_matrix is None or force_rebuild:
            self.user_item_matrix = self._build_user_item_matrix()

        if (self.user_similarity_matrix is None or force_rebuild) and not self.user_item_matrix.empty:
            self.user_similarity_matrix = self._calculate_user_similarity()

        if (self.item_similarity_matrix is None or force_rebuild) and not self.user_item_matrix.empty:
            self.item_similarity_matrix = self._calculate_item_similarity()

    def _calculate_content_score(self, property_data: Dict, user_preference: Dict) -> float:
        """Calculate content-based score with lenient scoring"""
        try:
            score = 0.0

            # 1. DISTANCE (30 points)
            distance = float(property_data.get('distance_from_campus', 999))
            preferred_distance = float(user_preference.get('preferred_distance', 2.0))

            if distance <= preferred_distance:
                distance_score = 1.0
            elif distance <= preferred_distance * 1.5:
                distance_score = 0.6
            else:
                distance_score = max(0, 1.0 - (distance - preferred_distance) / 5.0)

            score += distance_score * self.CONTENT_WEIGHTS['distance']

            # 2. COST (25 points)
            price = float(property_data.get('price', 0))
            budget_min = float(user_preference.get('budget_min', 0))
            budget_max = float(user_preference.get('budget_max', 999999))

            if budget_min <= price <= budget_max:
                cost_score = 1.0
            elif price < budget_min:
                cost_score = 0.7
            else:
                if budget_max > 0:
                    cost_score = max(0, 1.0 - (price - budget_max) / budget_max)
                else:
                    cost_score = 0.3

            score += cost_score * self.CONTENT_WEIGHTS['cost']

            # 3. SAFETY (15 points) - MORE LENIENT
            avg_rating = float(property_data.get('avg_rating', 3.0))

            if avg_rating >= 4.0:
                safety_score = 1.0
            elif avg_rating >= 3.0:
                safety_score = 0.6 + (avg_rating - 3.0) * 0.4
            else:
                safety_score = avg_rating / 5.0

            score += safety_score * self.CONTENT_WEIGHTS['safety']

            # 4. FACILITIES (10 points)
            user_amenities = set(user_preference.get('preferred_amenities', []))
            property_amenities = set(property_data.get('amenities', []))

            if user_amenities:
                amenity_match = len(user_amenities & property_amenities) / len(user_amenities)
            else:
                amenity_match = 1.0

            score += amenity_match * self.CONTENT_WEIGHTS['facilities']

            # 5. ROOM TYPE (10 points) - FIXED
            user_room_type = str(user_preference.get('room_type', 'Any'))
            property_room_type = str(property_data.get('room_type', 'Any'))

            if user_room_type == 'Any' or property_room_type == 'Any':
                room_score = 1.0
            elif user_room_type == property_room_type:
                room_score = 1.0
            else:
                room_score = 0.4

            score += room_score * self.CONTENT_WEIGHTS['room_type']

            # 6. MANAGEMENT (5 points) - MORE LENIENT
            rating_count = int(property_data.get('rating_count', 0))

            if rating_count >= 5:
                management_score = 1.0
            elif rating_count > 0:
                management_score = 0.2 + (rating_count / 5.0) * 0.8
            else:
                management_score = 0.0

            score += management_score * self.CONTENT_WEIGHTS['management']

            # 7. SOCIAL/GENDER (5 points) - FIXED
            user_gender_pref = str(user_preference.get('gender_preference', 'Any'))
            property_gender_restriction = str(property_data.get('gender_restriction', 'Any'))

            if user_gender_pref == 'Any' or property_gender_restriction == 'Any':
                social_score = 1.0
            elif user_gender_pref == property_gender_restriction:
                social_score = 1.0
            else:
                social_score = 0.3

            score += social_score * self.CONTENT_WEIGHTS['social']

            return max(0.0, min(100.0, round(score, 2)))

        except Exception as e:
            logger.error(f"Error calculating content score: {str(e)}")
            return 50.0

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
        """Detect algorithm type based on score breakdown"""
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
        adaptive_weights: Optional[Dict] = None,
        force_refresh: bool = False,
        include_rated: bool = False
    ) -> List[Dict]:
        """Get hybrid recommendations with ADAPTIVE WEIGHTS"""
        try:
            self._ensure_matrices(force_rebuild=force_refresh)

            if adaptive_weights is None:
                adaptive_weights = self.DEFAULT_WEIGHTS.copy()

            user_weight = adaptive_weights.get('user_based_cf', 0.4) * 100
            item_weight = adaptive_weights.get('item_based_cf', 0.3) * 100
            content_weight = adaptive_weights.get('content', 0.3) * 100

            logger.info(f"Using adaptive weights: User={user_weight}%, Item={item_weight}%, Content={content_weight}%")

            user_ratings = self.db.get_user_ratings(user_id)
            rated_property_ids = [r['property_id'] for r in user_ratings]

            all_properties = self.db.get_all_properties()

            if include_rated:
                candidate_properties = all_properties
            else:
                candidate_properties = [p for p in all_properties if p['id'] not in rated_property_ids]

            if not candidate_properties:
                logger.info(f"No candidate properties for user {user_id}")
                return []

            user_preference = self._get_fresh_user_preference(user_id, force_refresh=True)
            if not user_preference:
                user_preference = {}
                logger.info(f"No user preference found for user {user_id}, using defaults")

            recommendations = []

            for property_data in candidate_properties:
                try:
                    property_id = property_data['id']

                    amenities = self.db.get_property_amenities(property_id)
                    property_data['amenities'] = [a['amenity_id'] for a in amenities]

                    user_cf_score_normalized = self._get_user_based_score(user_id, property_id)
                    user_cf_score = user_cf_score_normalized * user_weight

                    item_cf_score_normalized = self._get_item_based_score(user_id, property_id, user_ratings)
                    item_cf_score = item_cf_score_normalized * item_weight

                    content_score_full = self._calculate_content_score(property_data, user_preference)
                    content_score = (content_score_full / 100.0) * content_weight

                    final_score = user_cf_score + item_cf_score + content_score
                    predicted_rating = (final_score / 100.0) * 5.0

                    breakdown = {
                        'user_based': round(user_cf_score, 2),
                        'item_based': round(item_cf_score, 2),
                        'content': round(content_score, 2)
                    }

                    detected_algorithm = self._detect_algorithm_from_scores(breakdown)
                    explanation = self._generate_explanation(breakdown, detected_algorithm)
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

            recommendations.sort(key=lambda x: x['total_score'], reverse=True)

            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations[:limit]

        except Exception as e:
            logger.error(f"Error in get_hybrid_recommendations for user {user_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_cold_start_recommendations(self, user_id: int, limit: int = 10, include_rated: bool = False) -> List[Dict]:
        """Cold start recommendations for new users"""
        try:
            user_preference = self._get_fresh_user_preference(user_id, force_refresh=True)

            all_properties = self.db.get_all_properties()

            if not include_rated:
                user_ratings = self.db.get_user_ratings(user_id)
                rated_property_ids = [r['property_id'] for r in user_ratings]
                all_properties = [p for p in all_properties if p['id'] not in rated_property_ids]

            recommendations = []

            for property_data in all_properties:
                amenities = self.db.get_property_amenities(property_data['id'])
                property_data['amenities'] = [a['amenity_id'] for a in amenities]

                avg_rating = float(property_data.get('avg_rating', 0))
                rating_count = int(property_data.get('rating_count', 0))

                popularity_score = min((avg_rating * 12) + (rating_count * 2), 100)

                content_score = 50.0
                if user_preference:
                    content_score = self._calculate_content_score(property_data, user_preference)

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
                return 0.5

            similar_users = self.find_similar_users(user_id, k=10)

            if not similar_users:
                return 0.5

            weighted_sum = 0.0
            similarity_sum = 0.0

            for sim_user in similar_users:
                sim_user_id = sim_user.user_id
                similarity = sim_user.similarity

                if property_id in self.user_item_matrix.columns:
                    rating = self.user_item_matrix.loc[sim_user_id, property_id]
                    if rating > 0:
                        weighted_sum += similarity * rating
                        similarity_sum += similarity

            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                return min(predicted_rating / 5.0, 1.0)

            return 0.5
        except Exception as e:
            logger.error(f"Error in _get_user_based_score: {str(e)}")
            return 0.5

    def _get_item_based_score(self, user_id: int, property_id: int, user_ratings: List[Dict]) -> float:
        """Get item-based CF score (0-1 normalized)"""
        try:
            if self.item_similarity_matrix is None or property_id not in self.item_similarity_matrix.index:
                return 0.5

            highly_rated = [r for r in user_ratings if float(r.get('rating', 0)) >= 4.0]

            if not highly_rated:
                return 0.5

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

        similarities = self.user_similarity_matrix.loc[user_id].sort_values(ascending=False)
        similarities = similarities[similarities.index != user_id]
        top_similar = similarities.head(k)

        similar_users = []
        for sim_user_id, similarity in top_similar.items():
            if similarity > 0:
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
                prop_users = set(self.user_item_matrix[property_id][self.user_item_matrix[property_id] > 0].index)
                sim_users = set(self.user_item_matrix[sim_prop_id][self.user_item_matrix[sim_prop_id] > 0].index)
                common_count = len(prop_users & sim_users)

                similar_properties.append(SimilarProperty(int(sim_prop_id), float(similarity), common_count))

        return similar_properties

    def get_user_based_recommendations(self, user_id: int, limit: int = 10, include_rated: bool = False) -> List[Dict]:
        """Pure user-based CF recommendations"""
        return self.get_hybrid_recommendations(
            user_id,
            limit,
            adaptive_weights={'user_based_cf': 0.5, 'item_based_cf': 0.3, 'content': 0.2},
            include_rated=include_rated
        )

    def get_item_based_recommendations(self, user_id: int, limit: int = 10, include_rated: bool = False) -> List[Dict]:
        """Pure item-based CF recommendations"""
        return self.get_hybrid_recommendations(
            user_id,
            limit,
            adaptive_weights={'user_based_cf': 0.1, 'item_based_cf': 0.5, 'content': 0.4},
            include_rated=include_rated
        )

    def predict_rating(self, user_id: int, property_id: int) -> float:
        """Predict rating for user-property pair"""
        recommendations = self.get_hybrid_recommendations(user_id, limit=100, force_refresh=True)

        for rec in recommendations:
            if rec['property_id'] == property_id:
                return rec['predicted_rating']

        return 3.0

    def clear_cache(self):
        """Clear cached matrices AND preferences"""
        self.user_item_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.cache = {}
        self.preference_cache = {}
        logger.info("Cache cleared (including preferences)")
