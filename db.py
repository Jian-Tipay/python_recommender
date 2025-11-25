import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.host = os.getenv("DB_HOST")
        self.database = os.getenv("DB_DATABASE")
        self.user = os.getenv("DB_USERNAME")
        self.password = os.getenv("DB_PASSWORD")
        self.port = int(os.getenv("DB_PORT", "3306"))
        self.connection = None

    def connect(self):
        try:
            if self.connection is None:
                self.connection = pymysql.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    port=self.port,
                    charset="utf8mb4",
                    cursorclass=pymysql.cursors.DictCursor
                )
            return self.connection
        except Exception as e:
            print(f"Error connecting to MySQL: {e}")
            raise

    def disconnect(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query, params=None):
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            print(f"Query error: {e}")
            raise

    def execute_single(self, query, params=None):
        try:
            conn = self.connect()
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchone()
        except Exception as e:
            print(f"Query error: {e}")
            raise
    
    def get_all_ratings(self):
        """Get all ratings from database"""
        query = """
            SELECT 
                rating_id,
                user_id,
                property_id,
                rating,
                created_at
            FROM ratings
            ORDER BY created_at DESC
        """
        return self.execute_query(query)
    
    def get_user_ratings(self, user_id):
        """Get all ratings by a specific user"""
        query = """
            SELECT 
                rating_id,
                user_id,
                property_id,
                rating,
                created_at
            FROM ratings
            WHERE user_id = %s
            ORDER BY created_at DESC
        """
        return self.execute_query(query, (user_id,))
    
    def get_property_ratings(self, property_id):
        """Get all ratings for a specific property"""
        query = """
            SELECT 
                rating_id,
                user_id,
                property_id,
                rating,
                created_at
            FROM ratings
            WHERE property_id = %s
            ORDER BY created_at DESC
        """
        return self.execute_query(query, (property_id,))
    
    def get_user_rated_properties(self, user_id):
        """Get list of property IDs rated by user"""
        query = """
            SELECT DISTINCT property_id
            FROM ratings
            WHERE user_id = %s
        """
        results = self.execute_query(query, (user_id,))
        return [r['property_id'] for r in results]
    
    def get_active_properties(self, exclude_ids=None):
        """Get all active properties"""
        query = """
            SELECT 
                id,
                title,
                address,
                price,
                distance_from_campus,
                landlord_id
            FROM properties
            WHERE is_active = 1
        """
        
        if exclude_ids:
            placeholders = ','.join(['%s'] * len(exclude_ids))
            query += f" AND id NOT IN ({placeholders})"
            return self.execute_query(query, tuple(exclude_ids))
        
        return self.execute_query(query)
    
    def get_property_details(self, property_id):
        """Get detailed information about a property"""
        query = """
            SELECT 
                id,
                landlord_id,
                title,
                description,
                address,
                price,
                distance_from_campus,
                is_active,
                created_at
            FROM properties
            WHERE id = %s
        """
        return self.execute_single(query, (property_id,))
    
    def get_user_preferences(self, user_id):
        """Get user preferences"""
        query = """
            SELECT 
                preference_id,
                user_id,
                preferred_distance,
                budget_min,
                budget_max,
                room_type,
                gender_preference
            FROM student_preferences
            WHERE user_id = %s
        """
        return self.execute_single(query, (user_id,))
    
    def get_property_amenities(self, property_id):
        """Get amenities for a property"""
        query = """
            SELECT a.amenity_id, a.amenity_name
            FROM property_amenities pa
            JOIN amenities a ON pa.amenity_id = a.amenity_id
            WHERE pa.property_id = %s
        """
        return self.execute_query(query, (property_id,))
    
    def get_rating_statistics(self):
        """Get rating statistics for the system"""
        query = """
            SELECT 
                COUNT(DISTINCT user_id) as total_users,
                COUNT(DISTINCT property_id) as total_properties,
                COUNT(*) as total_ratings,
                AVG(rating) as avg_rating,
                MIN(rating) as min_rating,
                MAX(rating) as max_rating
            FROM ratings
        """
        return self.execute_single(query)
    
    def __del__(self):
        """Cleanup connection on object destruction"""
        self.disconnect()
