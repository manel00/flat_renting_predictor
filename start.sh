#!/bin/bash

# Housing Price Prediction Application - Quick Start Script
# This script builds and starts the application using Docker Compose

echo "🏠 Housing Price Prediction Application"
echo "========================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed."
    echo "Please install Docker from https://www.docker.com/get-started"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: Docker Compose is not installed."
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"
echo ""

# Check if housing_data.csv exists
if [ ! -f "housing_data.csv" ]; then
    echo "❌ Error: housing_data.csv not found in the current directory"
    echo "Please make sure the data file is present"
    exit 1
fi

echo "✅ Data file found"
echo ""

# Stop any running containers
echo "🛑 Stopping any running containers..."
docker-compose down 2>/dev/null

echo ""
echo "🔨 Building and starting the application..."
echo "This may take a few minutes on first run..."
echo ""

# Build and start the containers
docker-compose up --build -d

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "✅ Application is running!"
    echo ""
    echo "📍 Access the application at:"
    echo "   Frontend:  http://localhost"
    echo "   Backend:   http://localhost:8000"
    echo "   API Docs:  http://localhost:8000/docs"
    echo ""
    echo "📊 To view logs:"
    echo "   docker-compose logs -f"
    echo ""
    echo "🛑 To stop the application:"
    echo "   docker-compose down"
    echo ""
else
    echo ""
    echo "❌ Error: Services failed to start"
    echo "Check logs with: docker-compose logs"
    exit 1
fi
