#! bash

# -----------------------------------------------------------------------------
# Script to test the dokerised property-value-predictor API
# Sends a POST request to the /predict endpoint with a JSON payload
# 
# Author: Tobias Holmes
# Created: 2025-07
# Description: Simple test script for the prediction API using curl.
# Requires 'test-payload.json' to be present in the same directory.
# -----------------------------------------------------------------------------

curl -X POST "http://127.0.0.1:8080/predict" \
-H "Content-Type: application/json" \
-d @test-payload.json