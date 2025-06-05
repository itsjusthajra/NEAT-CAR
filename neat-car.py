import math
import random
import sys
import os
import time

import neat
import pygame
import matplotlib.pyplot as plt
import numpy as np

# Constants
WIDTH = 1280
HEIGHT = 720

CAR_SIZE_X = 55    
CAR_SIZE_Y = 55

BORDER_COLOR = (100, 125, 49) # Color To Crash on Hit

current_generation = 0  # Generation counter
fitness_history = []  # Store fitness progression
average_fitness_history = []  # Store average fitness per generation
min_fitness_history = []  # Store minimum fitness per generation
std_fitness_history = []  # Store standard deviation of fitness per generation

# Track performance
collision_counts = []  # List to track collisions
distance_travelled = []  # List to track total distance travelled by cars
avg_speeds = []  # List to track the average speed of cars
time_alive = []  # Time cars remain alive in a generation
time_stuck = []  # Time cars are stuck in a generation
dead_cars_per_generation = []  # List to track the number of dead cars per generation

# Statistics to track the neural network's performance
activation_levels = []  # Activation levels of neurons
decision_probabilities = []  # Neural network decision probabilities

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('mycar.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.position = [654,660]  # Starting Position
        self.previous_positions = []  # Store previous positions
        self.stuck_counter = 0
        self.max_stuck_time = 60  # Max number of frames to be considered stuck
        
        self.angle = 0
        self.speed = 0
        self.speed_set = False
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]
        self.radars = []
        self.drawing_radars = []
        self.alive = True
        self.distance = 0
        self.time = 0
        self.path = []  # List to track car’s path
        self.time_alive = 0
        self.time_stuck = 0

    def is_stuck(self):
        if len(self.previous_positions) < 5:
            self.previous_positions.append(self.position[:])
            return False

        avg_distance = sum(
            math.sqrt(
                (self.previous_positions[i][0] - self.previous_positions[i - 1][0]) ** 2 +
                (self.previous_positions[i][1] - self.previous_positions[i - 1][1]) ** 2
            ) for i in range(1, len(self.previous_positions))
        ) / len(self.previous_positions)

        self.previous_positions.pop(0)

        if avg_distance < 2:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        return self.stuck_counter > self.max_stuck_time

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)  # Draw Sprite
        self.draw_radar(screen)  # Optionally draw sensors
        self.draw_path(screen)  # Optionally draw car's path

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            distance = radar[1]

            if distance < 50:
                color = (255, 0, 0)  # Red
            elif distance < 150:
                color = (255, 255, 0)  # Yellow
            else:
                color = (0, 255, 0)  # Green

            pygame.draw.line(screen, color, self.center, position, 1)
            pygame.draw.circle(screen, color, position, 5)

    def draw_path(self, screen):
        for point in self.path:
            pygame.draw.circle(screen, (0, 255, 0), point, 2)  # Visualize the path

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            if game_map.get_at((int(point[0]), int(point[1]))) == BORDER_COLOR:
                self.alive = False
                break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        map_width, map_height = game_map.get_size()

        while (
            0 <= x < map_width and 0 <= y < map_height and
            not game_map.get_at((x, y)) == BORDER_COLOR and
            length < 300
        ):
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        if 0 <= x < map_width and 0 <= y < map_height:
            dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
            self.radars.append([(x, y), dist])
        else:
            self.radars.append([(self.center[0], self.center[1]), 300])

    def update(self, game_map):
        if not self.speed_set:
            self.speed = 20
            self.speed_set = True

        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.distance += self.speed
        self.time += 1
        
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], WIDTH - 120)

        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(game_map)
        self.radars.clear()

        for d in range(-135, 136, 45):
            self.check_radar(d, game_map)

        # Record path
        self.path.append((int(self.center[0]), int(self.center[1])))

    def get_data(self):
        radars = self.radars
        return_values = [0] * 7
        for i, radar in enumerate(radars):
            return_values[i] = int(radar[1] / 30)
        return_values.append(self.speed / 100)
        return return_values

    def is_alive(self):
        return self.alive

    def get_reward(self):
        reward = self.distance / (CAR_SIZE_X / 2)
        for radar in self.radars:
            if radar[1] < 20:
                reward -= 5
        return reward

    def rotate_center(self, image, angle):
        rectangle = image.get_rect()
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rectangle.copy()
        rotated_rectangle.center = rotated_image.get_rect().center
        rotated_image = rotated_image.subsurface(rotated_rectangle).copy()
        return rotated_image

def plot_fitness_history():
    global fitness_history, average_fitness_history, min_fitness_history, std_fitness_history

    plt.figure()
    plt.plot(fitness_history, label="Max Fitness")
    plt.plot(average_fitness_history, label="Average Fitness")
    # plt.plot(min_fitness_history, label="Min Fitness")
    plt.plot(std_fitness_history, label="Fitness Standard Deviation")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Fitness Progression")
    plt.legend()
    plt.show()

def plot_dead_cars_per_generation():
    global dead_cars_per_generation

    plt.figure()
    plt.plot(dead_cars_per_generation, label="Dead Cars")
    plt.xlabel("Generations")
    plt.ylabel("Number of Dead Cars")
    plt.title("Dead Cars Per Generation")
    plt.legend()
    plt.show()

def run_simulation(genomes, config):
    global fitness_history, average_fitness_history, min_fitness_history, std_fitness_history
    global collision_counts, distance_travelled, avg_speeds, time_alive, time_stuck, dead_cars_per_generation
    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Times New Roman", 30)
    alive_font = pygame.font.SysFont("Times New Roman", 20)
    game_map = pygame.image.load('mark7.png').convert()

    global current_generation
    current_generation += 1

    counter = 0
    max_fitness = 0
    total_fitness = 0
    min_fitness = float('inf')

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                plot_fitness_history()
                plot_dead_cars_per_generation()
                sys.exit(0)

        for i, car in enumerate(cars):
            if car.is_alive():
                output = nets[i].activate(car.get_data())
                choice = output.index(max(output))
                activation_levels.append(output)  # Log neural network activation levels

                output_sum = sum(output)
                if output_sum > 0:
                  decision_probs = [prob / output_sum for prob in output]  # Probabilities of decision
                else:
                    decision_probs = [1.0 / len(output)] * len(output) # Probabilities of decision
                decision_probabilities.append(decision_probs)

                if choice == 0:
                    car.angle += 10
                elif choice == 1:
                    car.angle -= 10
                elif choice == 2:
                    if car.speed - 2 >= 12:
                        car.speed -= 2
                else:
                    car.speed += 2

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                car.update(game_map)

                if car.is_stuck():
                    genomes[i][1].fitness -= 10
                    car.alive = False
                else:
                    genomes[i][1].fitness += car.get_reward()
                    max_fitness = max(max_fitness, genomes[i][1].fitness)

                still_alive += 1

        if still_alive == 0:
            break

        counter += 1
        if counter == 60 * 20:
            break

        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        text = generation_font.render("Generation: " + str(current_generation), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.topright = (WIDTH - 20, 20)
        screen.blit(text, text_rect)

        text = alive_font.render("Alive: " + str(still_alive), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.topright = (WIDTH - 20, 60)
        screen.blit(text, text_rect)

        text = alive_font.render("Max Fitness: {:.2f}".format(max_fitness), True, (255, 255, 255))
        text_rect = text.get_rect()
        text_rect.topright = (WIDTH - 20, 100)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)

    # Track generation statistics
    fitness_values = [g.fitness for _, g in genomes]
    fitness_history.append(max(fitness_values))
    average_fitness_history.append(np.mean(fitness_values))
    min_fitness_history.append(np.min(fitness_values))
    std_fitness_history.append(np.std(fitness_values))

    # Track car performance metrics
    dead_cars = sum(1 for car in cars if not car.is_alive())
    dead_cars_per_generation.append(dead_cars)
    collision_counts.append(dead_cars)
    distance_travelled.append(sum(car.distance for car in cars))
    avg_speeds.append(np.mean([car.speed for car in cars]))
    time_alive.append(np.mean([car.time_alive for car in cars]))
    time_stuck.append(np.mean([car.time_stuck for car in cars]))

if __name__ == "__main__":
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(run_simulation, 20)
    plot_fitness_history()
    plot_dead_cars_per_generation()