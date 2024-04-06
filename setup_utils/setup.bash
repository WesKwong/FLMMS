#!/bin/bash

# Create a docker network for the FLMMS containers
./create_docker_net.bash

# Pull the necessary images for the FLMMS containers
./pull_image.bash