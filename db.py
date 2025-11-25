import mysql.connector
from mysql.connector import Error
import os
from dotenv import load_dotenv

load_dotenv()


class Database:
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.database = os.getenv('DB_DATABASE', 'slsu_boarding_house')
        self.user = os.getenv('DB_USERNAME', 'root')
        self.password = os.getenv('DB_PASSWORD', 'root')
        self.connection = None
    
    def connect(self):
        """Create database connection"""
        try:
            if self.connection is None or not self.connection.is_connected():
                self.connection = mysql.connector.connect(
                    host=self.host,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    charset='utf8mb4',
                    collation='utf8mb4_unicode_ci'
                )
            return self.connection
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.connection = None
    
    def test_connection(self):
        """Test if database connection works"""
        try:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Error as e:
            print(f"Connection test failed: {e}")
            raise
    
    def execute_query(self, query, params=None):
        """Execute a SELECT query and return results"""
        try:
            conn = self.connect()
            cursor = conn.cursor(dictionary=True)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            results = cursor.fetchall()
            cursor.close()
            
            return results
        except Error as e:
            print(f"Query execution error: {e}")
            raise
    
    def execute_single(self, query, params=None):
        """Execute a SELECT query and return single result"""
        try:
            conn = self.connect()
            cursor = conn.cursor(dictionary=True)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchone()
            cursor.close()
            
            return result
        except Error as e:
            print(f"Query execution error: {e}")
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