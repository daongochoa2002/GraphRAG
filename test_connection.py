#!/usr/bin/env python3
"""
Quick test script to verify Zilliz Cloud connection
"""

import os
from dotenv import load_dotenv
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

# Load environment variables
load_dotenv()

def test_zilliz_connection():
    """Test connection to Zilliz Cloud"""
    
    print("🧪 Testing Zilliz Cloud Connection...")
    
    # Get credentials from environment
    zilliz_uri = os.getenv('ZILLIZ_CLOUD_URI')
    zilliz_api_key = os.getenv('ZILLIZ_API_KEY')
    
    if not zilliz_uri or not zilliz_api_key:
        print("❌ Zilliz Cloud credentials not found in .env file")
        print("   Please set ZILLIZ_CLOUD_URI and ZILLIZ_API_KEY")
        return False
    
    try:
        # Connect to Zilliz Cloud
        print(f"🌩️  Connecting to: {zilliz_uri}")
        connections.connect(
            "default",
            uri=zilliz_uri,
            token=zilliz_api_key
        )
        
        # Test connection by getting server version
        version = utility.get_server_version()
        print(f"✅ Connected successfully! Server version: {version}")
        
        # List existing collections
        collections = utility.list_collections()
        print(f"📊 Existing collections: {collections}")
        
        # Test creating a simple collection
        test_collection_name = "connection_test"
        
        # Drop if exists
        if utility.has_collection(test_collection_name):
            utility.drop_collection(test_collection_name)
            print(f"🗑️  Dropped existing test collection")
        
        # Create test collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="test_vector", dtype=DataType.FLOAT_VECTOR, dim=128)
        ]
        
        schema = CollectionSchema(fields, "Test collection for connection")
        collection = Collection(test_collection_name, schema)
        
        print(f"✅ Created test collection: {test_collection_name}")
        
        # Clean up
        utility.drop_collection(test_collection_name)
        print(f"🧹 Cleaned up test collection")
        
        print("🎉 Zilliz Cloud connection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Zilliz Cloud Connection Test")
    print("=" * 40)
    
    success = test_zilliz_connection()
    
    if success:
        print("\n✅ All tests passed! You're ready to use Zilliz Cloud.")
        print("Next steps:")
        print("1. Set your OpenAI API key in .env file")
        print("2. Run: python upload_data.py")
        print("3. Run: python query_data.py")
    else:
        print("\n❌ Connection test failed. Please check your credentials.")
        print("Troubleshooting:")
        print("1. Verify your Zilliz Cloud URI and API key")
        print("2. Check your internet connection")
        print("3. Ensure your Zilliz cluster is active")

if __name__ == "__main__":
    main()
