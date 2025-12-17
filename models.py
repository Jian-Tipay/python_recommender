from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class Rating:
    """Rating model"""
    rating_id: int
    user_id: int
    property_id: int
    rating: float
    created_at: datetime
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            rating_id=data['rating_id'],
            user_id=data['user_id'],
            property_id=data['property_id'],
            rating=float(data['rating']),
            created_at=data['created_at']
        )


@dataclass
class Property:
    """Property model"""
    id: int
    landlord_id: int
    title: str
    address: str
    price: float
    distance_from_campus: Optional[float]
    description: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            id=data['id'],
            landlord_id=data['landlord_id'],
            title=data['title'],
            address=data.get('address', ''),
            price=float(data['price']),
            distance_from_campus=float(data['distance_from_campus']) if data.get('distance_from_campus') else None,
            description=data.get('description'),
            is_active=bool(data.get('is_active', 1)),
            created_at=data.get('created_at')
        )
    
    def to_dict(self):
        return {
            'property_id': self.id,
            'title': self.title,
            'address': self.address,
            'price': float(self.price),
            'distance_from_campus': float(self.distance_from_campus) if self.distance_from_campus else None
        }


@dataclass
class UserPreference:
    """User preference model"""
    preference_id: int
    user_id: int
    preferred_distance: Optional[float]
    budget_min: Optional[float]
    budget_max: Optional[float]
    room_type: str
    gender_preference: str
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            preference_id=data['preference_id'],
            user_id=data['user_id'],
            preferred_distance=float(data['preferred_distance']) if data.get('preferred_distance') else None,
            budget_min=float(data['budget_min']) if data.get('budget_min') else None,
            budget_max=float(data['budget_max']) if data.get('budget_max') else None,
            room_type=data.get('room_type', 'Any'),
            gender_preference=data.get('gender_preference', 'Any')
        )


@dataclass
class SimilarUser:
    """Similar user with similarity score"""
    user_id: int
    similarity: float
    common_ratings: int
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'similarity': round(self.similarity, 4),
            'common_ratings': self.common_ratings
        }


@dataclass
class SimilarProperty:
    """Similar property with similarity score"""
    property_id: int
    similarity: float
    common_users: int
    
    def to_dict(self):
        return {
            'property_id': self.property_id,
            'similarity': round(self.similarity, 4),
            'common_users': self.common_users
        }


@dataclass
class Recommendation:
    """Recommendation result"""
    property_id: int
    title: str
    address: str
    price: float
    distance_from_campus: Optional[float]
    predicted_rating: float
    confidence: float
    algorithm: str
    
    def to_dict(self):
        return {
            'property_id': self.property_id,
            'title': self.title,
            'address': self.address,
            'price': float(self.price),
            'distance_from_campus': float(self.distance_from_campus) if self.distance_from_campus else None,
            'predicted_rating': round(self.predicted_rating, 2),
            'confidence': round(self.confidence, 2),
            'algorithm': self.algorithm
        }


@dataclass
class RatingMatrix:
    """User-Item rating matrix"""
    users: List[int]
    properties: List[int]
    ratings: dict  # {(user_id, property_id): rating}
    
    def get_rating(self, user_id: int, property_id: int) -> Optional[float]:
        """Get rating for user-property pair"""
        return self.ratings.get((user_id, property_id))
    
    def get_user_ratings(self, user_id: int) -> dict:
        """Get all ratings by a user"""
        return {
            prop_id: rating 
            for (uid, prop_id), rating in self.ratings.items() 
            if uid == user_id
        }
    
    def get_property_ratings(self, property_id: int) -> dict:
        """Get all ratings for a property"""
        return {
            user_id: rating 
            for (user_id, pid), rating in self.ratings.items() 
            if pid == property_id
        }
    
    def has_user(self, user_id: int) -> bool:
        """Check if user exists in matrix"""
        return user_id in self.users
    
    def has_property(self, property_id: int) -> bool:
        """Check if property exists in matrix"""
        return property_id in self.properties