import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    
    print("🔍 Checking dependencies...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    else:
        print(f"✅ Python {sys.version.split()[0]} found")
    
    # Check required packages
    required_packages = [
        'pymilvus', 'transformers', 'torch', 'sentence_transformers', 
        'numpy', 'pandas', 'networkx', 'scikit-learn', 'python-dotenv',
        'tqdm', 'matplotlib', 'seaborn', 'accelerate', 'huggingface-hub'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} found")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages")
            return False
    
    return True

def check_environment():
    """Check environment configuration"""
    
    print("\n🔧 Checking environment configuration...")
    
    env_path = Path('.env')
    if not env_path.exists():
        print("❌ .env file not found")
        return False
    
    # Read .env file
    with open(env_path, 'r') as f:
        env_content = f.read()
    
    # Check for Zilliz Cloud credentials (no OpenAI needed for free model)
    if 'ZILLIZ_CLOUD_URI' in env_content and 'ZILLIZ_API_KEY' in env_content:
        print("✅ Zilliz Cloud configuration found")
        return True
    elif 'MILVUS_HOST' in env_content:
        print("✅ Local Milvus configuration found")
        return True
    else:
        print("❌ No valid Milvus/Zilliz configuration found")
        return False

def check_milvus():
    """Check Milvus/Zilliz Cloud connection"""
    
    print("\n🗃️  Checking vector database connection...")
    
    try:
        from pymilvus import connections, utility
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        # Check if using Zilliz Cloud or local Milvus
        zilliz_uri = os.getenv('ZILLIZ_CLOUD_URI')
        zilliz_api_key = os.getenv('ZILLIZ_API_KEY')
        
        if zilliz_uri and zilliz_api_key:
            # Try to connect to Zilliz Cloud
            print("🌩️  Attempting to connect to Zilliz Cloud...")
            connections.connect(
                "default",
                uri=zilliz_uri,
                token=zilliz_api_key
            )
            print("✅ Zilliz Cloud connection successful")
        else:
            # Try to connect to local Milvus
            print("🏠 Attempting to connect to local Milvus...")
            connections.connect("default", host="localhost", port="19530")
            print("✅ Local Milvus connection successful")
        
        # Test connection by getting server version
        if utility.get_server_version():
            return True
        else:
            print("❌ Failed to get server version")
            return False
            
    except Exception as e:
        print(f"❌ Vector database connection failed: {e}")
        if 'zilliz_uri' in locals() and zilliz_uri:
            print("   Please check your Zilliz Cloud credentials in .env file")
        else:
            print("   Please ensure Milvus is running on localhost:19530")
            print("   You can start Milvus using: docker-compose up -d")
        return False

def check_dataset():
    """Check if dataset exists"""
    
    print("\n📊 Checking dataset...")
    
    dataset_path = Path('datasets/icews14/train.txt')
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        return False
    else:
        # Check if dataset has content
        with open(dataset_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 0:
            print(f"✅ Dataset found with {len(lines)} entries")
            return True
        else:
            print("❌ Dataset file is empty")
            return False

def main():
    """Main health check function"""
    
    print("🏥 Graph RAG System Health Check")
    print("=" * 40)
    
    checks = [
        ("Dependencies", check_dependencies),
        ("Environment", check_environment),
        ("Milvus", check_milvus),
        ("Dataset", check_dataset)
    ]
    
    all_passed = True
    results = {}
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results[check_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results[check_name] = False
            all_passed = False
    
    print("\n" + "=" * 40)
    print("📋 Health Check Summary")
    print("=" * 40)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:15} {status}")
    
    if all_passed:
        print("\n🎉 All checks passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python upload_data.py' to upload data")
        print("2. Run 'python query_data.py' to start querying")
        print("3. Run 'python evaluate_performance.py' to evaluate")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above before proceeding.")
        print("\nTroubleshooting:")
        if not results.get("Dependencies", True):
            print("- Install missing dependencies: pip install -r requirements.txt")
        if not results.get("Environment", True):
            print("- Update your OpenAI API key in .env file")
        if not results.get("Milvus", True):
            print("- Start Milvus: docker-compose up -d")
        if not results.get("Dataset", True):
            print("- Ensure dataset files are in datasets/icews14/ directory")

if __name__ == "__main__":
    main()
