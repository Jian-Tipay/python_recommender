<<<<<<< HEAD
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.connection = pymysql.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USERNAME'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_DATABASE'),
            port=int(os.getenv('DB_PORT', 3306)),
            cursorclass=pymysql.cursors.DictCursor
        )

    def test_connection(self):
        """Test database connection"""
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            return cursor.fetchone()

    def get_all_ratings(self):
        """Get all ratings"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT user_id, property_id, rating, created_at
                FROM ratings
                ORDER BY user_id, property_id
            """)
            return cursor.fetchall()

    def get_user_ratings(self, user_id):
        """Get ratings for a specific user"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT property_id, rating, created_at
                FROM ratings
                WHERE user_id = %s
                ORDER BY created_at DESC
            """, (user_id,))
            return cursor.fetchall()

    def get_active_properties(self):
        """Get all active properties (alias method)"""
        return self.get_all_properties()

    def get_all_properties(self):
        """Get all active properties with details - INCLUDES gender_restriction"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    p.id,
                    p.title,
                    p.price,
                    p.distance_from_campus,
                    p.room_type,
                    p.address,
                    p.gender_restriction,
                    COALESCE(AVG(r.rating), 0) as avg_rating,
                    COUNT(r.rating_id) as rating_count
                FROM properties p
                LEFT JOIN ratings r ON p.id = r.property_id
                WHERE p.is_active = 1
                GROUP BY p.id, p.title, p.price, p.distance_from_campus,
                         p.room_type, p.address, p.gender_restriction
                ORDER BY p.id
            """)
            return cursor.fetchall()

    def get_property_amenities(self, property_id):
        """Get amenities for a property"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT a.amenity_id, a.amenity_name
                FROM property_amenities pa
                JOIN amenities a ON pa.amenity_id = a.amenity_id
                WHERE pa.property_id = %s
            """, (property_id,))
            return cursor.fetchall()

    def get_user_preference(self, user_id):
        """Get user preferences - INCLUDES gender_preference"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
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
                LIMIT 1
            """, (user_id,))
            result = cursor.fetchone()

            if result:
                # Get preferred amenities
                cursor.execute("""
                    SELECT amenity_id
                    FROM preferred_amenities
                    WHERE preference_id = %s
                """, (result['preference_id'],))
                result['preferred_amenities'] = [row['amenity_id'] for row in cursor.fetchall()]

            return result

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
=======
import pymysql
import os
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.connection = pymysql.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USERNAME'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_DATABASE'),
            port=int(os.getenv('DB_PORT', 3306)),
            cursorclass=pymysql.cursors.DictCursor
        )
    
    def test_connection(self):
        """Test database connection"""
        with self.connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            return cursor.fetchone()
    
    def get_all_ratings(self):
        """Get all ratings"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT user_id, property_id, rating
                FROM ratings
                ORDER BY user_id, property_id
            """)
            return cursor.fetchall()
    
    def get_user_ratings(self, user_id):
        """Get ratings for a specific user"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT property_id, rating
                FROM ratings
                WHERE user_id = %s
                ORDER BY property_id
            """, (user_id,))
            return cursor.fetchall()
    
    def get_active_properties(self):
        """Get all active properties (alias method)"""
        return self.get_all_properties()
    
    def get_all_properties(self):
        """Get all active properties with details"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    p.id,
                    p.title,
                    p.price,
                    p.distance_from_campus,
                    p.room_type,
                    p.address,
                    COALESCE(AVG(r.rating), 0) as avg_rating,
                    COUNT(r.rating_id) as rating_count
                FROM properties p
                LEFT JOIN ratings r ON p.id = r.property_id
                WHERE p.is_active = 1
                GROUP BY p.id
                ORDER BY p.id
            """)
            return cursor.fetchall()
    
    def get_property_amenities(self, property_id):
        """Get amenities for a property"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT a.amenity_id, a.amenity_name
                FROM property_amenities pa
                JOIN amenities a ON pa.amenity_id = a.amenity_id
                WHERE pa.property_id = %s
            """, (property_id,))
            return cursor.fetchall()
    
    def get_user_preference(self, user_id):
        """Get user preferences"""
        with self.connection.cursor() as cursor:
            cursor.execute("""
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
                LIMIT 1
            """, (user_id,))
            result = cursor.fetchone()
            
            if result:
                # Get preferred amenities
                cursor.execute("""
                    SELECT amenity_id
                    FROM preferred_amenities
                    WHERE preference_id = %s
                """, (result['preference_id'],))
                result['preferred_amenities'] = [row['amenity_id'] for row in cursor.fetchall()]
            
            return result
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

>>>>>>> 86a58891953078be2aad369e8c9396b449a91fdc
