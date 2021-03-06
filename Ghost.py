import Items
import pygame
import random
from operator import *
import math
from Constants import *

class Ghost:
    def __init__(self, x, y, color, scatter_coord, personality):

        #if ghosts are disabled, trap them in box
        if turnOffGhosts: 
            x = house_x
            y = house_y

        # CONSTANTS
        self.offset = block_size * 2
        self.size = 24
        self.base_color = color
        self.step_len = block_size / 16  # normal movement speed
        self.slow_step = block_size / 24  # movement speed when turned blue
        self.step = self.step_len
        self.personality = personality
        self.scatter_time = 5
        self.chase_time = 20

        # MOVEMENT
        self.scatter_coord = scatter_coord
        self.COORD_DIR = {0: [1, 0], 1: [0, 1], 2: [-1, 0], 3: [0, -1]}
        self.look_dir = 3
        self.move_dir = 3

        # LOCATION
        self.array_coord = [x, y]
        self.x = x * block_size + block_size / 2  # px
        self.y = y * block_size + block_size / 2  # px

        # SETUP VARS
        self.mode = "house"
        self.blue = False
        self.respawn_time = 3

        # TIMERS
        # Initialise multiple timers to avoid accidental collisions
        self.turn_timer = 0
        self.respawn_timer = 0
        self.timer = 0
        self.blue_timer = 0

    def move(self, player, maze, display_width, tick_counter, blinky_coord):
        def find_distance(a_pos, b_pos):
            a = pow(abs(a_pos[0] - b_pos[0]), 2)
            b = pow(abs(a_pos[1] - b_pos[1]), 2)
            return math.sqrt(a + b)

        def find_closest(facing, target_pos):
            return_dir = facing
            next_pos = list(map(add, self.array_coord, self.COORD_DIR[facing]))
            dir_min = find_distance(next_pos, target_pos)
            # check left turn
            if maze.can_move(self, left_turn(facing)):
                next_pos = list(map(add, self.array_coord, self.COORD_DIR[left_turn(facing)]))
                next_dir = find_distance(next_pos, target_pos)
                if next_dir < dir_min:
                    dir_min = next_dir
                    return_dir = left_turn(facing)
            # check right turn
            if maze.can_move(self, right_turn(facing)):
                next_pos = list(map(add, self.array_coord, self.COORD_DIR[right_turn(facing)]))
                next_dir = find_distance(next_pos, target_pos)
                if next_dir < dir_min:
                    return_dir = right_turn(facing)
            return return_dir

        def left_turn(facing):
            return abs((facing - 1) % 4)

        def right_turn(facing):
            return abs((facing + 1) % 4)

        # Set step length based on current state
        step = self.step_len
        if self.blue:
            if self.blue_timer >= player.power_time:
                self.blue = False
            else:
                self.blue_timer += 1
                step = self.slow_step
        if self.mode == "dead":
            self.step = self.step_len * 2
            step = self.step

        if self.mode == "normal" or self.mode == "dead":
            # Update current position in array
            self.array_coord = [int(self.x / block_size), int(self.y / block_size)]

            # Only try changing direction if within bounds of maze array
            if block_size < self.x < display_width - block_size:
                my_row = int(self.y / block_size)
                my_col = int(self.x / block_size)
                player_row = int(player.y / block_size)
                player_col = int(player.x / block_size)
                player_dir = 0

                # FRIGHTENED MODE
                if self.blue:
                    # Check whether the player is visible from this ghost's perspective
                    # If they are on the same row or column, check all tiles in between
                    # If there are no walls, the ghost can see the player
                    see_player = False
                    if my_row == player_row or my_col == player_col:
                        wall = False  # flag for whether there is an obstruction between ghost and player
                        if my_col == player_col and my_row == player_row:
                            wall = True
                        elif my_row == player_row:
                            if my_col > player_col:
                                player_dir = LEFT
                                for i in range(0, my_col - player_col):
                                    if maze.maze_array[my_row][i + player_col] == 1:
                                        wall = True
                            elif player_col == my_col:
                                wall = True
                            else:
                                player_dir = RIGHT
                                for i in range(0, player_col - my_col):
                                    if maze.maze_array[my_row][i + my_col] == 1:
                                        wall = True
                        elif my_col == player_col:
                            if my_row > player_row:
                                player_dir = UP
                                for i in range(0, my_row - player_row):
                                    if maze.maze_array[i + player_row][my_col] == 1:
                                        wall = True
                            elif player_row == my_row:
                                wall = True
                            else:
                                player_dir = DOWN
                                for i in range(0, player_row - my_row):
                                    if maze.maze_array[i + my_row][my_col] == 1:
                                        wall = True
                        if not wall:
                            see_player = True

                    # Only attempt turn if more than 1 tick since last turn
                    if self.turn_timer > 2:
                        # Run away from the player if it is visible
                        # If it is able to continue in the direction it is facing it will
                        # do so, so long as it does not go towards the player
                        if see_player:
                            if self.look_dir == player_dir or not maze.can_move(self, self.look_dir):
                                self.look_dir = random.choice([left_turn(left_turn(player_dir)),
                                                               left_turn(player_dir), right_turn(player_dir)])
                                self.turn_timer = 0
                        # if player not visible, pick a random movement direction
                        else:
                            self.look_dir = random.choice([self.move_dir, left_turn(self.move_dir),
                                                           right_turn(self.move_dir)])
                            self.turn_timer = 0

                # set target position based on current behaviour
                if self.mode == "normal":
                    # Immediately exit house
                    if self.array_coord == [house_x, house_y]:
                        target_coord = [house_x, house_y-2]
                    elif self.array_coord in([house_x, house_y-1], [house_x, house_y-2]) and self.move_dir == UP:
                        target_coord = [house_x-1, house_y-2]
                    # Scatter
                    elif (tick_counter / 60) % (self.chase_time + self.scatter_time) < self.scatter_time:
                        target_coord = self.scatter_coord
                    # Chase Pac-Man
                    else:
                        target_coord = player.array_coord
                        if self.personality == "speedy":
                            target_coord = \
                                [player.array_coord[0] + player.COORD_DIR[player.move_dir][0] * 2,
                                 player.array_coord[1] + player.COORD_DIR[player.move_dir][1] * 2]
                        if self.personality == "pokey" and find_distance(self.array_coord, player.array_coord) < 3:
                            target_coord = self.scatter_coord
                        if self.personality == "bashful":
                            target_coord = \
                                [player.array_coord[0] + player.COORD_DIR[player.move_dir][0] * 2,
                                player.array_coord[1] + player.COORD_DIR[player.move_dir][1] * 2]
                            vect = list(map(sub, target_coord, blinky_coord))
                            target_coord = list(map(add, target_coord, vect))
                # if dead, move back to ghost house
                elif self.mode == "dead":
                    target_coord = [house_x, house_y]

                # move towards target, only attempt a turn at an intersection
                if step < self.x % block_size < block_size - step \
                        and step < self.y % block_size < block_size - step and self.turn_timer > 2:
                    if maze.can_move(self, left_turn(self.look_dir)) \
                            or maze.can_move(self, right_turn(self.look_dir)):
                        self.look_dir = find_closest(self.look_dir, target_coord)
                        self.turn_timer = 0
                    if not maze.can_move(self, self.look_dir):
                        self.look_dir = random.choice([left_turn(self.move_dir), right_turn(self.move_dir)])
                        self.turn_timer = 0

                # change move direction to match look direction if possible
                if self.look_dir != self.move_dir:
                    if maze.can_move(self, self.look_dir):
                        self.move_dir = self.look_dir
                    # if in a dead end, flip direction
                    if not (maze.can_move(self, self.move_dir)) \
                            and not (maze.can_move(self, left_turn(self.move_dir))) \
                            and not (maze.can_move(self, right_turn(self.move_dir))):
                        self.look_dir = left_turn(left_turn(self.move_dir))
                        self.move_dir = self.look_dir

                # do movement
                if maze.can_move(self, self.move_dir):
                    self.x += step * self.COORD_DIR[self.move_dir][0]
                    self.y += step * self.COORD_DIR[self.move_dir][1]

            # If outside maze, keep moving forwards until wrapped to the other side of the screen
            else:
                if self.move_dir == LEFT:
                    self.x -= self.step_len
                    maze.center(self, "y", self.y)
                if self.move_dir == RIGHT:
                    self.x += self.step_len
                    maze.center(self, "y", self.y)
                # screen wrap
                if self.x < -self.size:
                    self.x = int(0.5*block_size + MapSizeX*block_size) #display_width
                if self.x > self.size + display_width:
                    self.x = -int(0.5*block_size) #-self.size

            # respawn if way found back to house
            if self.mode == "dead" and self.array_coord == [house_x, house_y]:
                self.mode = "normal"

            self.turn_timer += 1

        # Ghost stays in the house and paces left and right
        elif self.mode == "house":
            if self.look_dir == DOWN or self.look_dir == UP:
                self.look_dir = random.choice([LEFT, RIGHT])
                self.move_dir = self.look_dir
            if not (maze.can_move(self, self.move_dir)):
                self.look_dir = left_turn(left_turn(self.move_dir))
                self.move_dir = self.look_dir
            self.x += step * self.COORD_DIR[self.move_dir][0]

    def draw(self, display, player, tick_counter):
        def draw_body(col):
            pygame.draw.ellipse(display, col, (self.x - self.size / 2, self.y + self.offset - self.size / 2,
                                                    self.size, self.size * 0.95))
            pygame.draw.rect(display, col, (self.x - self.size / 2, self.y + self.offset,
                                                 self.size, self.size / 4))

            # alternate wobble shape
            if 0 < tick_counter % 20 < 10:
                pygame.draw.ellipse(display, col, (
                    self.x - self.size / 2, self.y + self.size / 6 + self.offset - 1, self.size / 3, self.size / 3))
                pygame.draw.ellipse(display, col, (
                    self.x - self.size / 6, self.y + self.size / 6 + self.offset - 1, self.size / 3, self.size / 3))
                pygame.draw.ellipse(display, col, (
                self.x + self.size / 6, self.y + self.size / 6 + self.offset - 1, self.size / 3, self.size / 3))
            else:
                pygame.draw.ellipse(display, col, (
                    self.x - self.size / 6 * 2, self.y + self.size / 6 + self.offset - 1, self.size / 3, self.size / 3))
                pygame.draw.ellipse(display, col, (
                    self.x, self.y + self.size / 6 + self.offset - 1, self.size / 3, self.size / 3))

        def draw_eyes(move_dir):
            eye_width = self.size / 3
            eye_height = eye_width * 3 / 2
            pupil_diam = eye_width * 3 / 4
            eye_separation = self.size * 0.1
            y_pos = self.y - eye_height / 2 + self.offset

            x_off = 0
            y_off = 0
            if move_dir == RIGHT:
                x_off = 1
            elif move_dir == LEFT:
                x_off = -1
            elif move_dir == UP:
                y_off = -1
            elif move_dir == DOWN:
                y_off = 1

            # eye whites
            pygame.draw.ellipse(display, (255, 255, 255),
                                (self.x - eye_width - (eye_separation / 2) + x_off, y_pos + y_off,
                                 eye_width, eye_height))
            pygame.draw.ellipse(display, (255, 255, 255),
                                (self.x + (eye_separation / 2) + x_off, y_pos + y_off,
                                 eye_width, eye_height))

            # eye pupils
            pygame.draw.circle(display, (0, 0, 0), (round(self.x - eye_width / 2 - eye_separation / 2 + x_off * 2),
                               round(y_pos + eye_height / 2 + y_off * 2)), round(pupil_diam / 2))
            pygame.draw.circle(display, (0, 0, 0), (round(self.x + eye_width / 2 + eye_separation / 2 + x_off * 2),
                               round(y_pos + eye_height / 2 + y_off * 2)), round(pupil_diam / 2))

        if self.mode != "dead":
            if self.blue and player.powered_up:
                # blink blue and white in the last 2 seconds of power up time
                if 0 < self.blue_timer % 40 < 20 \
                        and self.blue_timer + (2 * 60) >= player.power_time:
                    color = (200, 200, 255)  # very light blue
                else:
                    color = (50, 50, 200)  # dark blue
            else:
                color = self.base_color
            draw_body(color)
            draw_eyes(self.move_dir)

        else:
            draw_eyes(self.move_dir)

    def collide(self, player):
        dist_x = abs(self.x - player.x)
        dist_y = abs(self.y - player.y)
        touch_distance = self.size / 2

        return dist_x < touch_distance and dist_y < touch_distance and self.mode != "dead"
