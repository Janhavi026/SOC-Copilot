import mysql.connector
from mysql.connector import Error
import configparser
import os

def test_database_connection():
    """
    Reads database configuration from config.ini, connects to the database,
    and performs a simple query to verify the connection and table access.
    """
    print("--- Database Connection Debugger ---")

    # --- 1. Read Configuration File ---
    try:
        config_path = 'config.ini'
        if not os.path.exists(config_path):
            print(f"❌ ERROR: Configuration file not found.")
            print(f"   - Expected Location: {os.path.abspath(config_path)}")
            return

        config = configparser.ConfigParser()
        config.read(config_path)
        db_config = dict(config['DATABASE'])
        print("✅ SUCCESS: `config.ini` file read successfully.")
        print(f"   - Host: {db_config.get('host')}")
        print(f"   - Database: {db_config.get('database')}")
        print(f"   - User: {db_config.get('user')}")

    except Exception as e:
        print(f"❌ ERROR: Could not read or parse the [DATABASE] section in config.ini.")
        print(f"   - Details: {e}")
        return

    # --- 2. Attempt to Connect to the Database ---
    connection = None
    try:
        print("\nAttempting to connect to the database...")
        connection = mysql.connector.connect(**db_config)
        
        if connection.is_connected():
            print("✅ SUCCESS: Connection to the MySQL database was successful!")
        else:
            print("❌ FAILED: Connection attempt did not succeed (no error thrown).")
            return
            
    except Error as e:
        print(f"❌ FAILED: Could not connect to the database.")
        print(f"   - Error Code: {e.errno}")
        print(f"   - SQLSTATE: {e.sqlstate}")
        print(f"   - Message: {e.msg}")
        print("\n   Possible Causes:")
        print("     - Firewall is blocking the connection on port 3306.")
        print("     - Database server is not running.")
        print("     - Incorrect host IP, username, or password in config.ini.")
        return
    
    # --- 3. Verify Table Access ---
    cursor = None
    try:
        print("\nAttempting to query the 'currentrequeststatus' table...")
        cursor = connection.cursor()
        
        # This query checks if the table exists and can be accessed.
        cursor.execute("SELECT COUNT(*) FROM `currentrequeststatus`;")
        record_count = cursor.fetchone()[0]
        
        print(f"✅ SUCCESS: The 'currentrequeststatus' table exists and contains {record_count} records.")
        
    except Error as e:
        print(f"❌ FAILED: Could not query the 'currentrequeststatus' table.")
        print(f"   - Error Code: {e.errno}")
        print(f"   - Message: {e.msg}")
        print("\n   Possible Causes:")
        print("     - The table does not exist in the 'soar' database.")
        print("     - The user 'root' does not have permission to access this table.")
        
    finally:
        if connection and connection.is_connected():
            if cursor:
                cursor.close()
            connection.close()
            print("\nDatabase connection closed.")


if __name__ == '__main__':
    test_database_connection()
