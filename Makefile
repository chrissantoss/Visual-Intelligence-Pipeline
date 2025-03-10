.PHONY: setup test test-unit test-integration test-performance test-system load-test batch-process download-samples clean

# Default target
all: setup test

# Setup
setup:
	@echo "Setting up environment..."
	@mkdir -p logs data/sample test_results

# Run all tests
test: test-unit test-integration test-performance

# Run unit tests
test-unit:
	@echo "Running unit tests..."
	@python -m pytest app/tests/unit -v

# Run integration tests
test-integration:
	@echo "Running integration tests..."
	@python -m pytest app/tests/integration -v

# Run performance tests
test-performance:
	@echo "Running performance tests..."
	@python -m pytest app/tests/integration/test_performance.py -v

# Run system tests
test-system:
	@echo "Running system tests..."
	@python test_system.py

# Run load tests
load-test:
	@echo "Running load tests..."
	@python load_test.py --num-clients 5 --num-requests 10

# Download sample images
download-samples:
	@echo "Downloading sample images..."
	@python download_sample_images.py --num-images 10

# Run batch processing
batch-process: download-samples
	@echo "Running batch processing..."
	@python batch_process.py --input-dir data/sample --output-dir test_results/batch

# Run the API server
run-api:
	@echo "Running API server..."
	@uvicorn app.main:app --reload

# Run with Docker
docker-build:
	@echo "Building Docker image..."
	@docker-compose build

docker-up:
	@echo "Starting Docker containers..."
	@docker-compose up -d

docker-down:
	@echo "Stopping Docker containers..."
	@docker-compose down

docker-test:
	@echo "Running tests in Docker..."
	@docker-compose exec api pytest

# Clean up
clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__ .pytest_cache
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete 