#!/bin/bash

# Graph RAG Setup Script
echo "🚀 Setting up Graph RAG Application..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Check if Docker is installed for Milvus
if command -v docker &> /dev/null; then
    echo "🐳 Docker found. Setting up Milvus..."
    
    # Download Milvus docker-compose file
    if [ ! -f "docker-compose.yml" ]; then
        echo "📥 Downloading Milvus docker-compose configuration..."
        wget https://github.com/milvus-io/milvus/releases/download/v2.3.4/milvus-standalone-docker-compose.yml -O docker-compose.yml
    fi
    
    # Start Milvus
    echo "🚀 Starting Milvus..."
    docker-compose up -d
    
    # Wait for Milvus to be ready
    echo "⏳ Waiting for Milvus to be ready..."
    sleep 30
    
    echo "✅ Milvus is running on localhost:19530"
else
    echo "⚠️  Docker not found. Please install Docker to run Milvus locally."
    echo "   Alternatively, you can use a cloud Milvus instance and update the .env file."
fi

# Check if .env file exists and has API key
if [ -f ".env" ]; then
    if grep -q "your_openai_api_key_here" .env; then
        echo "⚠️  Please update your OpenAI API key in the .env file"
    else
        echo "✅ Environment configuration found"
    fi
else
    echo "❌ .env file not found. Please create it from the template."
fi

echo ""
echo "🎉 Setup completed!"
echo ""
echo "Next steps:"
echo "1. Update your OpenAI API key in .env file"
echo "2. Run 'python upload_data.py' to process and upload data"
echo "3. Run 'python query_data.py' to start querying"
echo "4. Run 'python evaluate_performance.py' to evaluate the system"
echo ""
echo "📚 See README.md for detailed instructions"
