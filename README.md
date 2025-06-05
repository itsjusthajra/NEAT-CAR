# NEAT-CAR

## Overview

This project implements an AI-driven car racing simulation using Pygame and the NEAT (NeuroEvolution of Augmenting Topologies) library. Cars learn to navigate a track (mark7.png) using a neural network that evolves over generations. The simulation tracks fitness, collisions, distance traveled, and other metrics, visualized through plots.

## Prerequisites

Python 3.x
Required libraries: pygame, neat-python, matplotlib, numpy

Install dependencies using:
pip install pygame neat-python matplotlib numpy

## Files

neat-car.py: The main simulation script.
mycar.png: The car sprite (a red car, as shown in the provided image).
mark7.png: The track image (a black track on a green background with a checkered start/finish line, as shown in the provided image).
config.txt: NEAT configuration file (must be configured for your setup).

## How to Run

Ensure all required files (mycar.png, mark7.png, config.txt) are in the same directory as neat-car.py.

### Run the simulation:

python neat-car.py

The simulation will run for 20 generations, displaying the cars navigating the track. At the end, fitness and dead car plots are generated.

## Creating New Tracks

To create a new track:

Design the Track: Use an image editor to create a track similar to mark7.png. The track should be black (RGB: (0, 0, 0)), the borders green (RGB: (100, 125, 49)), and the background green. Include a checkered start/finish line (black and white pattern).

Save the Track: Save the track image as a PNG file (e.g., new_track.png) with dimensions matching the simulation window (1280x720 by default).
Update the Code: In main.py, modify the line loading the track image in the run_simulation function:
game_map = pygame.image.load('new_track.png').convert()

## Positioning Cars on New Tracks

Cars are positioned using the position attribute in the Car class, which is a list [x, y] representing the top-left corner of the car sprite. The car size is 55x55 pixels (CAR_SIZE_X, CAR_SIZE_Y).

### Steps to Position Cars
Identify Coordinates: Open your new track image in an image editor to find the coordinates of the starting position (e.g., on the start/finish line). Ensure the position keeps the car within the track boundaries, accounting for the car's size.

### Update Starting Position: In the Car class __init__ method, modify the self.position:

self.position = [x, y]  # Replace x, y with your coordinates

For example, if the start/finish line on your new track is at coordinates (300, 500):

self.position = [300, 500]

Verify Position: Run the simulation to ensure the car starts on the track and can move without immediate collisions.
Example with Provided Track (mark7.png)
The provided track has a checkered start/finish line at the bottom center.
The default position [654, 660] places the car on this line.
For a new track, adjust these coordinates to match your start/finish line.

## Simulation Features

Neural Network: NEAT evolves the car's behavior to avoid borders and maximize distance traveled.
Fitness Tracking: Tracks max, average, min, and standard deviation of fitness across generations.
Performance Metrics: Logs collisions, distance traveled, average speed, time alive, and time stuck.
Visualization: Plots fitness progression and dead cars per generation using Matplotlib.

## Customization
Window Size: Modify WIDTH and HEIGHT to match your track dimensions.
Car Size: Adjust CAR_SIZE_X and CAR_SIZE_Y if using a different car sprite.
Border Color: Update BORDER_COLOR to match your track's border color.
Simulation Duration: Change the number of generations in population.run(run_simulation, 20).

## Notes

The car sprite (mycar.png) must be in the same directory as the script.
The track image must have the same border color as defined in BORDER_COLOR for collision detection to work.
