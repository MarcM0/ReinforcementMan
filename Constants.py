block_size = 30
half_block_size = block_size / 2
offset = block_size * 2
pellet_size = 6
power_pellet_size = 12
house_x = 14
house_y = 13
spawn_x = 14
spawn_y = 17
pellet_score = 10
power_pellet_score = 50
ghost_scores = [200, 400, 800, 1600]
# Fruit images from https://static.wikia.nocookie.net/pacman/images/2/25/Fruits_Points.png/revision/latest?cb=20210921001546
fruit_images = ["FruitImgs/cherry.png", "FruitImgs/strawberry.png", "FruitImgs/orange.png", "FruitImgs/apple.png", "FruitImgs/melon.png", "FruitImgs/starship.png", "FruitImgs/bell.png", "FruitImgs/key.png"]
fruit_scores = [100, 300, 500, 700, 1000, 2000, 3000, 5000]
fruit_time = 10
life_points = 10000

#Quick Toggles
fastMode = False #No longer human playable, increases speed of game to absolute limits
neatFrameShow = 512 #show every x frames when in fastMode, try to have this be a power of 2
showFPS = False #shows fps, use for testing, prints clutter and slow down program

#movement constants
RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
