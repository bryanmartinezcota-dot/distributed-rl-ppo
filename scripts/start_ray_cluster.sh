#!/bin/bash

echo "Starting local Ray cluster head node..."

# Stop any existing ray instances
ray stop

# Start the head node with a dashboard for monitoring
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=16

echo "Cluster is up! Monitor rollouts at http://localhost:8265"
