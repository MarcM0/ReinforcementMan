import random
from re import L, X
from tkinter import Y
from turtle import down, right
from Maze import Maze
from Items import *
from Ghost import Ghost
from Constants import *
from operator import *
import math
from Pac_Man import Pac_Man
import numpy as np
import copy
from collections import deque
import random

#returns grid index for 1D array
def gridToArray(x,y,rowSize=MapSizeX):
    return y*rowSize+x

#Process the inputs for the nead model 
#just passing in everything (SET inputs IN neatConfig to 913)
def NaiveNeatHelper(pacman, maze, ghosts, pellets, power_pellets, fruit):
    #converting the inputs to a 1D array to pass into NEAT
    inputs = np.array(maze.maze_array)
    inputs = inputs.flatten()
    
    #tell ai about fruit
    inputs = np.append(inputs,[fruit.here, fruit.x, fruit.y])

    #tell ai about power pellets
    for power_pellet in power_pellets:
        inputs = np.append(inputs,[power_pellet.here, power_pellet.x, power_pellet.y])
    
    #tell ai about ghosts (boo)
    for ghost in ghosts.values():
        inputs = np.append(inputs,[ghost.x, ghost.y, ghost.blue, ghost.move_dir, (ghost.mode == "normal"), (pacman.power_time - ghost.blue_timer)])
    
    #tell ai about pacman
    inputs = np.append(inputs,[pacman.x, pacman.y, pacman.move_dir, pacman.powered_up, (pacman.power_time - pacman.timer),(pacman.lives)])

    return inputs    

#multi group multi destination djikstra, returns shortest path to closest of each group
def multiDest(start, objectiveGroups, blocked, mazeArray, notAllowed = [],ghostMoveBlocks=None,exceptions=[]):

    start = queueNode(start[0], start[1])

    paths = [-1]*len(objectiveGroups)
    movingTowards = 0

    queue = list()
    visited = set()

    visited.add((start.x,start.y))

    queue.append(start)

    dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))

    while queue:

        curr = queue.pop(0)

        allFound = True
        for group in objectiveGroups:
            if(group != -1 and group):
                allFound = False
                break
        
        if(allFound):
            if(ghostMoveBlocks != None): return paths,movingTowards
            return paths
        
        for index in range(len(objectiveGroups)):
            if [curr.x, curr.y] in objectiveGroups[index] and paths[index] == -1:
                index2 = objectiveGroups[index].index([curr.x, curr.y])
                path = []
                getPath(curr, path)
                paths[index]=path
                lastNode = path[len(path)-2]
                if ghostMoveBlocks != None and ghostMoveBlocks[index2] == [lastNode.x,lastNode.y]:
                    movingTowards = 1

        #check adjacents and add to queue
        for testX, testY in dirs:
                x = (testX+curr.x)%len(mazeArray[0])
                y = (testY+curr.y)%len(mazeArray)

                if(((not mazeArray[y][x] in blocked) or [x,y] in exceptions) and (not [x,y] in notAllowed) and (not (x,y) in notAllowed)):
                    nextCoord = queueNode(x,y, curr)
                    if (nextCoord.x,nextCoord.y) not in visited:
                        queue.append(nextCoord)
                        visited.add((nextCoord.x,nextCoord.y))

    if(ghostMoveBlocks != None): return paths,movingTowards
    return paths


def rotatingCameraNeatHelper(pacman, maze, ghosts, pellets, power_pellets, fruit):

    #get grid position
    pac_manX = int(((pacman.x-block_size/2.0)/block_size))%len(maze.maze_array[0]) #get pacman location
    pac_manY = int(((pacman.y-block_size/2.0)/block_size))%len(maze.maze_array) 
    truePos = [pac_manX,pac_manY]
    
    #get pacman true position (not rounded)
    x = pacman.x-block_size/2.0 
    y = pacman.y-block_size/2.0
    truePos2 = [(((x)/block_size)),  (((y)/block_size))]

    if block_size < pacman.x < MapSizeX* block_size - block_size:
        canmove = np.array([[maze.can_move(pacman, UP)==True,maze.can_move(pacman, RIGHT)==True],[maze.can_move(pacman, LEFT)==True,maze.can_move(pacman, DOWN)==True]])
    else:
        canmove = np.array([[False,False],[False,False]])

    foundBlue = False
    for ghost in ghosts.values():
        if(ghost.blue): foundBlue = True
        break

    ghostObjs = []
    blueAndFruitObjs = []
    pelletsObjs = []
    powerPelletObjs = []
    ghostMoveBlocks = []
    exceptions = []
    if(forceStuck and neatMode and not (turnOffGhosts and truePos!=[9,13]) and not foundBlue): #make dead end seem like ghost
        exceptions = [[9,12]]
        ghostObjs = [[9,12]]
        ghostMoveBlocks = [[9,12]]

    upNotAllowed =  [((pac_manX+1)%len(maze.maze_array[0]),(pac_manY)%len(maze.maze_array)),
                    ((pac_manX-1)%len(maze.maze_array[0]),(pac_manY)%len(maze.maze_array)),
                    ((pac_manX)%len(maze.maze_array[0]),(pac_manY+1)%len(maze.maze_array))]
    rightNotAllowed=[((pac_manX)%len(maze.maze_array[0]),(pac_manY+1)%len(maze.maze_array)),
                    ((pac_manX-1)%len(maze.maze_array[0]),(pac_manY)%len(maze.maze_array)),
                    ((pac_manX)%len(maze.maze_array[0]),(pac_manY-1)%len(maze.maze_array))]
    leftNotAllowed= [((pac_manX)%len(maze.maze_array[0]),(pac_manY+1)%len(maze.maze_array)),
                    ((pac_manX)%len(maze.maze_array[0]),(pac_manY-1)%len(maze.maze_array)),
                    ((pac_manX+1)%len(maze.maze_array[0]),(pac_manY)%len(maze.maze_array))]
    downNotAllowed= [((pac_manX+1)%len(maze.maze_array[0]),(pac_manY  )%len(maze.maze_array)),
                    ((pac_manX-1)%len(maze.maze_array[0]),(pac_manY)%len(maze.maze_array)),
                    ((pac_manX)%len(maze.maze_array[0]),(pac_manY-1)%len(maze.maze_array))]
    
     #dijkstra distances
    closeGhosts = np.array([[12,12],[12,12]],dtype=float)
    ghostApproaching = np.array([[0,0],[0,0]],dtype=float)
    for ghost in ghosts.values():
            if (not (turnOffGhosts)) and ghost.mode == "normal" and (not ghost.blue or (ghost.blue_timer + (2) >= pacman.power_time)):
                x = ghost.x-block_size/2.0
                y = ghost.y-block_size/2.0
                testTile = [int((x/block_size)),  int((y/block_size))]
                if(testTile == truePos):
                    diffx = testTile[0]-truePos2[0]
                    diffy = testTile[1]-truePos2[1]
                    if(diffx != 0):
                        if diffx<0:
                            testTile[0] = testTile[0]-1
                        else:
                            testTile[0] = testTile[0]+1
                    else:
                        if diffy<0:
                            testTile[1] = testTile[1]-1
                        else:
                            testTile[1] = testTile[1]+1
                    exceptions.append(testTile)
                testTile[0] = testTile[0]%len(maze.maze_array[0])
                testTile[1] = testTile[1]%len(maze.maze_array)
                destTile = testTile.copy()
                if(ghost.move_dir == UP):
                        destTile[1] = (destTile[1]-1)%len(maze.maze_array)
                if(ghost.move_dir == RIGHT):
                        destTile[0] = (destTile[0]+1)%len(maze.maze_array[0])
                if(ghost.move_dir == LEFT):
                        destTile[0] = (destTile[0]-1)%len(maze.maze_array[0])
                if(ghost.move_dir == DOWN):
                        destTile[1] = (destTile[1]+1)%len(maze.maze_array)
                ghostObjs.append(testTile)  
                ghostMoveBlocks.append(destTile)

    if pacman.frameMod60%15==0 or ((pacman.lastghostObjs!=ghostObjs or pacman.lasttruePos!=truePos or pacman.ghostMoveBlocksLast!=ghostMoveBlocks or not np.array_equal(canmove, pacman.canmoveLast))):
    
        if (ghostObjs):
            path,movingTowards = multiDest([pac_manX, pac_manY], [ghostObjs], [1],maze.maze_array, upNotAllowed,ghostMoveBlocks,exceptions)
            if path[0] != -1: closeGhosts[0,0] = len(path[0] )/5
            ghostApproaching[0,0] = movingTowards
            path,movingTowards = multiDest([pac_manX, pac_manY], [ghostObjs], [1],maze.maze_array, rightNotAllowed,ghostMoveBlocks,exceptions)
            if path[0]  != -1: closeGhosts[0,1] = len(path[0] )/5
            ghostApproaching[0,1] = movingTowards
            path,movingTowards = multiDest([pac_manX, pac_manY], [ghostObjs], [1],maze.maze_array, leftNotAllowed,ghostMoveBlocks,exceptions)
            if path[0]  != -1: closeGhosts[1,0] = len(path[0] )/5
            ghostApproaching[1,0] = movingTowards
            path,movingTowards = multiDest([pac_manX, pac_manY], [ghostObjs], [1],maze.maze_array, downNotAllowed,ghostMoveBlocks,exceptions)
            if path[0] != -1: closeGhosts[1,1] = len(path[0] )/5
            ghostApproaching[1,1] = movingTowards
            
        exceptions = []
        #dijkstra distances
        for ghost in ghosts.values():
                if (not (turnOffGhosts)) and ghost.mode == "normal" and (ghost.blue and not (ghost.blue_timer + (2) >= pacman.power_time)):
                    x = ghost.x-block_size/2.0
                    y = ghost.y-block_size/2.0
                    testTile = [int((x/block_size)),  int((y/block_size))]
                    if(testTile == truePos):
                        diffx = testTile[0]-truePos2[0]
                        diffy = testTile[1]-truePos2[1]
                        if(diffx != 0):
                            if diffx<0:
                                testTile[0] = testTile[0]-1
                            else:
                                testTile[0] = testTile[0]+1
                        else:
                            if diffy<0:
                                testTile[1] = testTile[1]-1
                            else:
                                testTile[1] = testTile[1]+1
                        exceptions.append(testTile)
                    testTile[0] = testTile[0]%len(maze.maze_array[0])
                    testTile[1] = testTile[1]%len(maze.maze_array)
                    blueAndFruitObjs.append(testTile)      
        
        if fruit.here:
            testTile = fruit.array_coord.copy()
            if(testTile == truePos):
                diffx = testTile[0]-truePos2[0]
                diffy = testTile[1]-truePos2[1]
                if(diffx != 0):
                    if diffx<0:
                        testTile[0] = testTile[0]-1
                    else:
                        testTile[0] = testTile[0]+1
                else:
                    if diffy<0:
                        testTile[1] = testTile[1]-1
                    else:
                        testTile[1] = testTile[1]+1
                exceptions.append(testTile)
            testTile[0] = testTile[0]%len(maze.maze_array[0])
            testTile[1] = testTile[1]%len(maze.maze_array)
            blueAndFruitObjs.append(testTile)
        
        for power_pellet in power_pellets:
                if power_pellet.here:
                    testTile = power_pellet.array_coord.copy()
                    if(testTile == truePos):
                        diffx = testTile[0]-truePos2[0]
                        diffy = testTile[1]-truePos2[1]
                        if(diffx != 0):
                            if diffx<0:
                                testTile[0] = testTile[0]-1
                            else:
                                testTile[0] = testTile[0]+1
                        else:
                            if diffy<0:
                                testTile[1] = testTile[1]-1
                            else:
                                testTile[1] = testTile[1]+1
                        exceptions.append(testTile)
                    testTile[0] = testTile[0]%len(maze.maze_array[0])
                    testTile[1] = testTile[1]%len(maze.maze_array)
                    powerPelletObjs.append(testTile)
        
        for pellet in pellets:
            if pellet.here:
                    testTile = pellet.array_coord.copy()
                    if(testTile == truePos):
                        diffx = testTile[0]-truePos2[0]
                        diffy = testTile[1]-truePos2[1]
                        if(diffx != 0):
                            if diffx<0:
                                testTile[0] = testTile[0]-1
                            else:
                                testTile[0] = testTile[0]+1
                        else:
                            if diffy<0:
                                testTile[1] = testTile[1]-1
                            else:
                                testTile[1] = testTile[1]+1
                        exceptions.append(testTile)
                    testTile[0] = testTile[0]%len(maze.maze_array[0])
                    testTile[1] = testTile[1]%len(maze.maze_array)
                    pelletsObjs.append(testTile)
        
        closeBlueGhosts = np.array([[12,12],[12,12]],dtype=float)
        closePellets = np.array([[12,12],[12,12]],dtype=float)
        closePowerPellets = np.array([[12,12],[12,12]],dtype=float)
        
        objGroups =  [pelletsObjs,powerPelletObjs,blueAndFruitObjs]
        closeGroups =  [closePellets,closePowerPellets,closeBlueGhosts]
        if pelletsObjs or powerPelletObjs or blueAndFruitObjs:
            paths = multiDest([pac_manX, pac_manY], objGroups, [1,2],maze.maze_array, upNotAllowed+ghostObjs,None,exceptions)
            for path,close in zip(paths,closeGroups):
                if path != -1 and canmove[0,0]: close[0,0] = len(path)/5
            paths = multiDest([pac_manX, pac_manY], objGroups, [1,2],maze.maze_array, rightNotAllowed+ghostObjs,None,exceptions)
            for path,close in zip(paths,closeGroups):
                if path != -1 and canmove[0,1]: close[0,1] = len(path)/5
            paths = multiDest([pac_manX, pac_manY], objGroups, [1,2],maze.maze_array, leftNotAllowed+ghostObjs,None,exceptions)
            for path,close in zip(paths,closeGroups):
                if path != -1 and canmove[1,0]: close[1,0] = len(path)/5
            paths = multiDest([pac_manX, pac_manY], objGroups, [1,2],maze.maze_array, downNotAllowed+ghostObjs,None,exceptions)
            for path,close in zip(paths,closeGroups):
                if path != -1 and canmove[1,1]: close[1,1] = len(path)/5
    else: #use cached data
        closeGhosts = pacman.closeGhostsLast.copy()
        closeBlueGhosts = pacman.closeBlueGhostsLast.copy()
        closePowerPellets = pacman.closePowerPelletsLast.copy()
        closePellets = pacman.closePelletsLast.copy()
        ghostApproaching = pacman.lastghostAproach.copy()

    #cache data so we don't recalaculate later
    pacman.lastghostObjs = ghostObjs.copy()
    pacman.lastghostAproach = ghostApproaching.copy()
    pacman.lasttruePos = truePos.copy()
    pacman.closeGhostsLast = closeGhosts.copy()
    pacman.closeBlueGhostsLast = closeBlueGhosts.copy()
    pacman.closePowerPelletsLast = closePowerPellets.copy()
    pacman.closePelletsLast = closePellets.copy()
    pacman.ghostMoveBlocksLast = ghostMoveBlocks.copy()
    pacman.canmoveLast = canmove.copy()

    #find nearest wall and intersection
    maxDistW = 6
    closeWalls2 = np.array([[maxDistW,maxDistW],[maxDistW,maxDistW]],dtype=float)
    closeInters2 = np.array([[maxDistW,maxDistW],[maxDistW,maxDistW]],dtype=float)
    dirsToTest = np.array([[(0,-1),(1,0)],[(-1,0),(0,1)]],dtype=tuple)
    for x in range(2):
        for y in range(2):
            direction = dirsToTest[x,y]
            testBlock= [truePos[0] + direction[0], truePos[1] + direction[1]]
            while(testBlock[0]<MapSizeX and testBlock[0]>0 and testBlock[1]<MapSizeY and testBlock[1]>0):
                #check for walls
                if(closeWalls2[x,y]==maxDistW and ( (maze.maze_array[testBlock[1]][testBlock[0]] in [1,2])) ):
                    
                    if(direction[0] == 0):
                        closeWalls2[x,y] = abs(testBlock[1]-truePos[1])/5
                    else:
                        closeWalls2[x,y] = abs(testBlock[0]-truePos[0])/5
                    break

                if(closeInters2[x,y]==maxDistW):
                    outs = 0
                    adjacents =  [[testBlock[0] ,testBlock[1]-1 ], [ testBlock[0]+1,testBlock[1] ] , [ testBlock[0]-1,testBlock[1] ], [testBlock[0],testBlock[1]+1] ]
                    for adj in adjacents:
                        if((adj[0]<MapSizeX and adj[0]>0 and adj[1]<MapSizeY and adj[1]>0) and not (maze.maze_array[adj[1]][adj[0]] in [1,2])):
                            outs+=1
                        if(outs>2):
                            if(direction[0] == 0):
                                closeInters2[x,y] = abs(testBlock[1]-truePos[1])/5
                            else:
                                closeInters2[x,y] = abs(testBlock[0]-truePos[0])/5
                            break

                testBlock[0] = testBlock[0] + direction[0]
                testBlock[1] = testBlock[1] + direction[1]


    closeGhosts2 = np.array([[6,6],[6,6]],dtype=float)
    for ghost in ghosts.values():
        if((turnOffGhosts) or (ghost.blue and not (ghost.blue_timer + (30) >= pacman.power_time)) or ghost.mode != "normal"): continue
        x = ghost.x-block_size/2.0
        y = ghost.y-block_size/2.0
        testTile = [(x/block_size),  (y/block_size)]
        distx = testTile[0]-truePos2[0]
        disty = testTile[1]-truePos2[1]
        totalDist = abs(distx)+abs(disty)
        totalDist = totalDist/5
      
        if(disty<0 and abs(distx) < 2 and totalDist<closeGhosts2[0,0] and totalDist<closeWalls2[0,0]+0.2):
            closeGhosts2[0,0] = totalDist
        if(disty>0 and abs(distx) < 2 and totalDist<closeGhosts2[1,1] and totalDist<closeWalls2[1,1]+0.2):
            closeGhosts2[1,1] = totalDist
        if(distx<0 and abs(disty) < 2 and totalDist<closeGhosts2[1,0] and totalDist<closeWalls2[1,0]+0.2):
            closeGhosts2[1,0] =totalDist
        if(distx>0 and abs(disty) < 2 and totalDist<closeGhosts2[0,1] and totalDist<closeWalls2[0,1]+0.2):
            closeGhosts2[0,1] = totalDist
    
    if(forceStuck and neatMode and not (turnOffGhosts and truePos!=[9,13]) and not foundBlue):
        testTile = [9,12]
        distx = testTile[0]-truePos2[0]
        disty = testTile[1]-truePos2[1]
        totalDist = abs(distx)+abs(disty)
        totalDist = totalDist/5
    
        if(disty<0 and abs(distx) < 2 and totalDist<closeGhosts2[0,0] and totalDist<closeWalls2[0,0]+0.2):
            closeGhosts2[0,0] = totalDist
        if(disty>0 and abs(distx) < 2 and totalDist<closeGhosts2[1,1] and totalDist<closeWalls2[1,1]+0.2):
            closeGhosts2[1,1] = totalDist
        if(distx<0 and abs(disty) < 2 and totalDist<closeGhosts2[1,0] and totalDist<closeWalls2[1,0]+0.2):
            closeGhosts2[1,0] =totalDist
        if(distx>0 and abs(disty) < 2 and totalDist<closeGhosts2[0,1] and totalDist<closeWalls2[0,1]+0.2):
            closeGhosts2[0,1] = totalDist

    #sensing
    closeBlueGhosts2 = np.array([[6,6],[6,6]],dtype=float)
    for ghost in ghosts.values():
        if((turnOffGhosts) or (not ghost.blue or (ghost.blue_timer + (30) >= pacman.power_time)) or ghost.mode != "normal"): continue
        x = ghost.x-block_size/2.0 #get the top left corner
        y = ghost.y-block_size/2.0
        testTile = [(x/block_size),  (y/block_size)]
        distx = testTile[0]-truePos2[0]
        disty = testTile[1]-truePos2[1]
        totalDist = abs(distx)+abs(disty)
        totalDist = totalDist/5
        
        if(disty<0 and abs(distx) < 2 and canmove[0,0] and totalDist<closeBlueGhosts2[0,0] and totalDist<closeWalls2[0,0]+0.2):
            closeBlueGhosts2[0,0] = totalDist
        if(disty>0 and abs(distx) < 2 and canmove[1,1] and totalDist<closeBlueGhosts2[1,1] and totalDist<closeWalls2[1,1]+0.2):
            closeBlueGhosts2[1,1] = totalDist
        if(distx<0 and abs(disty)  < 2 and canmove[1,0] and totalDist<closeBlueGhosts2[1,0] and totalDist<closeWalls2[1,0]+0.2):
            closeBlueGhosts2[1,0] = totalDist
        if(distx>0 and abs(disty) < 2 and canmove[0,1] and totalDist<closeBlueGhosts2[0,1] and totalDist<closeWalls2[0,1]+0.2):
            closeBlueGhosts2[0,1] = totalDist

    #add fruit to blue ghosts
    if fruit.here:
        distx = fruit.array_coord[0]-truePos2[0]
        disty = fruit.array_coord[1]-truePos2[1]
        totalDist = abs(distx)+abs(disty)
        totalDist = totalDist/5

        if(disty<0 and abs(distx) < 2 and canmove[0,0] and totalDist<closeBlueGhosts2[0,0]) and totalDist<closeWalls2[0,0]+0.2:
            closeBlueGhosts2[0,0] = totalDist
        if(disty>0 and abs(distx) < 2 and canmove[1,1] and totalDist<closeBlueGhosts2[1,1]) and totalDist<closeWalls2[1,1]+0.2:
            closeBlueGhosts2[1,1] = totalDist
        if(distx<0 and abs(disty) < 2 and canmove[1,0] and totalDist<closeBlueGhosts2[1,0]) and totalDist<closeWalls2[1,0]+0.2:
            closeBlueGhosts2[1,0] = totalDist
        if(distx>0 and abs(disty) < 2 and canmove[0,1] and totalDist<closeBlueGhosts2[0,1]) and totalDist<closeWalls2[0,1]+0.2:
            closeBlueGhosts2[0,1] = totalDist

    closePellets2 = np.array([[6,6],[6,6]],dtype=float)
    for pellet in pellets:
        if not pellet.here: continue
        distx = pellet.array_coord[0]-truePos2[0]
        disty = pellet.array_coord[1]-truePos2[1]
        totalDist = abs(distx)+abs(disty)
        totalDist = totalDist/5
        
        if(disty<0 and abs(distx) < 2 and canmove[0,0] and totalDist<closePellets2[0,0] and totalDist<closeWalls2[0,0]+0.2):
            closePellets2[0,0] =totalDist
        if(disty>0 and abs(distx) < 2 and canmove[1,1] and totalDist<closePellets2[1,1] and totalDist<closeWalls2[1,1]+0.2):
            closePellets2[1,1] =totalDist
        if(distx<0 and abs(disty) < 2 and canmove[1,0] and totalDist<closePellets2[1,0] and totalDist<closeWalls2[1,0]+0.2):
            closePellets2[1,0] = totalDist
        if(distx>0 and abs(disty)< 2 and canmove[0,1] and totalDist<closePellets2[0,1] and totalDist<closeWalls2[0,1]+0.2):
            closePellets2[0,1] = totalDist

    closePowerPellets2 = np.array([[6,6],[6,6]],dtype=float)
    for powerPellet in power_pellets:
        if not powerPellet.here: continue
        distx = powerPellet.array_coord[0]-truePos2[0]
        disty = powerPellet.array_coord[1]-truePos2[1]
        totalDist = abs(distx)+abs(disty)
        totalDist = totalDist/5
        
        if(disty<0 and abs(distx) < 2 and canmove[0,0] and totalDist<closePowerPellets2[0,0] and totalDist<closeWalls2[0,0]+0.2):
            closePowerPellets2[0,0] = totalDist
        if(disty>0 and abs(distx)< 2 and canmove[1,1] and totalDist<closePowerPellets2[1,1] and totalDist<closeWalls2[1,1]+0.2):
            closePowerPellets2[1,1] = totalDist
        if(distx<0 and abs(disty) < 2 and canmove[1,0] and totalDist<closePowerPellets2[1,0] and totalDist<closeWalls2[1,0]+0.2):
            closePowerPellets2[1,0] = totalDist
        if(distx>0 and abs(disty)< 2 and canmove[0,1] and totalDist<closePowerPellets2[0,1] and totalDist<closeWalls2[0,1]+0.2):
            closePowerPellets2[0,1] = totalDist

    pacman.closeGhostsLast = closeGhosts.copy()
    pacman.closeBlueGhostsLast = closeBlueGhosts.copy()
    pacman.closePowerPelletsLast = closePowerPellets.copy()
    pacman.closePelletsLast = closePellets.copy()
    rotateDir = pacman.move_dir

    unrotatedCloseghosts = closeGhosts.reshape(-1)
    unrotatedGhostApproaching = ghostApproaching.reshape(-1)

    if(rotateCamera):
        if rotateDir == RIGHT:
            closeGhosts = np.rot90(closeGhosts, 1).reshape(-1)
            closeBlueGhosts = np.rot90(closeBlueGhosts, 1).reshape(-1)
            closePellets = np.rot90(closePellets, 1).reshape(-1)
            closePowerPellets = np.rot90(closePowerPellets, 1).reshape(-1)
            closeGhosts2 = np.rot90(closeGhosts2, 1).reshape(-1)
            closeBlueGhosts2 = np.rot90(closeBlueGhosts2, 1).reshape(-1)
            closePellets2 = np.rot90(closePellets2, 1).reshape(-1)
            closePowerPellets2 = np.rot90(closePowerPellets2, 1).reshape(-1)
            ghostApproaching = np.rot90(ghostApproaching, 1).reshape(-1)
            closeInters2 = np.rot90(closeInters2, 1).reshape(-1)
            canmove = np.rot90(canmove, 1).reshape(-1)
            
        elif rotateDir == DOWN:
            closeGhosts = np.rot90(closeGhosts, 2).reshape(-1)
            closeBlueGhosts = np.rot90(closeBlueGhosts, 2).reshape(-1)
            closePellets = np.rot90(closePellets, 2).reshape(-1)
            closePowerPellets = np.rot90(closePowerPellets, 2).reshape(-1)
            closeGhosts2 = np.rot90(closeGhosts2, 2).reshape(-1)
            closeBlueGhosts2 = np.rot90(closeBlueGhosts2, 2).reshape(-1)
            closePellets2 = np.rot90(closePellets2, 2).reshape(-1)
            closePowerPellets2 = np.rot90(closePowerPellets2, 2).reshape(-1)
            ghostApproaching = np.rot90(ghostApproaching, 2).reshape(-1)
            closeInters2 = np.rot90(closeInters2, 2).reshape(-1)
            canmove = np.rot90(canmove, 2).reshape(-1)
        elif rotateDir == LEFT:
            closeGhosts = np.rot90(closeGhosts, 3).reshape(-1)
            closeBlueGhosts = np.rot90(closeBlueGhosts, 3).reshape(-1)
            closePellets = np.rot90(closePellets, 3).reshape(-1)
            closePowerPellets = np.rot90(closePowerPellets, 3).reshape(-1)
            closeGhosts2 = np.rot90(closeGhosts2, 3).reshape(-1)
            closeBlueGhosts2 = np.rot90(closeBlueGhosts2, 3).reshape(-1)
            closePellets2 = np.rot90(closePellets2, 3).reshape(-1)
            closePowerPellets2 = np.rot90(closePowerPellets2, 3).reshape(-1)
            ghostApproaching = np.rot90(ghostApproaching, 3).reshape(-1)
            closeInters2 = np.rot90(closeInters2, 3).reshape(-1)
            canmove = np.rot90(canmove, 3).reshape(-1)
        else:
            closeGhosts = closeGhosts.reshape(-1)
            closeBlueGhosts = closeBlueGhosts.reshape(-1)
            closePellets = closePellets.reshape(-1)
            closePowerPellets = closePowerPellets.reshape(-1)
            closeGhosts2 = closeGhosts2.reshape(-1)
            closeBlueGhosts2 = closeBlueGhosts2.reshape(-1)
            closePellets2 = closePellets2.reshape(-1)
            closePowerPellets2 = closePowerPellets2.reshape(-1)
            ghostApproaching = ghostApproaching.reshape(-1)
            closeInters2 = closeInters2.reshape(-1)
            canmove = canmove.reshape(-1)
    
    inputs = np.concatenate(([abs(pacman.move_dir - pacman.lastMoveDir)!=2,pacman.pelletRatio,pacman.framesNotMoving,pacman.framesNotScoring/60,pacman.allGhostsOut],closeInters2,closeGhosts,closeBlueGhosts,closePellets,closePowerPellets,closeGhosts2,closeBlueGhosts2,closePellets2,closePowerPellets2,canmove,ghostApproaching)).astype(np.float32, copy=False)
    return inputs,unrotatedCloseghosts, unrotatedGhostApproaching
    
#nead model controller
def modelNeat(pacman, maze, ghosts, pellets, power_pellets, fruit):
    
    #Get the inputs
    inputs = rotatingCameraNeatHelper(pacman, maze, ghosts, pellets, power_pellets, fruit)

    #pass inputs into the neural network
    outputs = [pacman.net.activate(inputsUP)[0],pacman.net.activate(inputsRIGHT)[0],pacman.net.activate(inputsLEFT)[0],pacman.net.activate(inputsDOWN)[0]]
    nextMove = 0
    
    # interpret net output 
    if(oneOutput):
        if(len(outputs)!=1): print("oneOutput on but not 1 output")
        if(outputs[0] >= 0):
                    nextMove = pacman.move_dir #yes I know these don't match, im trying something
        else:
                    nextMove = pacman.look_dir + 1
    
    elif(wacky2Output):
        if(len(outputs)!=2): print("wacky2Outputs on but not 2 outputs")
        if(outputs[0]>0):
            nextMove = pacman.move_dir 
        else:
            if(outputs[1] >= 0):
                    nextMove = pacman.move_dir - 1
            else:
                if ((pacman.look_dir-pacman.move_dir)%4) == 1:
                    nextMove = pacman.look_dir + 1
                else:
                    nextMove = pacman.move_dir + 1


    elif(len(outputs) == 2):
            #interpret net output 
            if(abs(outputs[0]) > abs(outputs[1])):
                axis = 0
            else:
                axis = 1
            
            if(outputs[axis]>0):
                if(axis == 0):
                    nextMove = pacman.move_dir
                else:
                    nextMove = pacman.move_dir + 1
            else:
                if(axis == 0):
                    nextMove = pacman.move_dir + 2
                else:
                    nextMove =  pacman.move_dir - 1
        
    elif(len(outputs) == 3):
            #interpret net output 
            if(outputs[2]>0):
                axis = 0
            else:
                axis = 1
            
            if(outputs[axis]>0):
                if(axis == 0):
                    nextMove = pacman.move_dir
                else:
                    nextMove = pacman.move_dir + 1
            else:
                if(axis == 0):
                    nextMove = pacman.move_dir + 2
                else:
                    nextMove = pacman.move_dir -1
    
    elif(len(outputs) == 4):
        maxInd = -1
        maxVal = -9999
        for i in range(4):
            if(outputs[i]>maxVal):
                maxVal = outputs[i]
                maxInd = i

        FourDirs = [pacman.move_dir,pacman.move_dir+1,pacman.move_dir+3,pacman.move_dir+2]
        nextMove = FourDirs[maxInd]

    nextMove = nextMove%4 

    fourDirs2=[UP,RIGHT,LEFT,DOWN]
    for Dir, closeghost, approaching in zip(fourDirs2,unrotatedCloseghosts2,unrotatedGhostApproaching):
        if(nextMove == Dir and approaching and closeghost<2/5): 
            pacman.penalty+=suicidePenalty
            break
            
    return nextMove

def printMaze(maze):
    for y in range(len(maze)):
        row = " "
        for x in range(len(maze[0])):
            row += str(maze[y][x]) + " "
        print(row)

#random moves -> no smart
def dummy(pac_man, maze, ghosts, pellets, power_pellets, fruit):
    if(random.randint(0,10) == 0):
        return random.randint(0,4)
    return pac_man.move_dir

#a custom hit ditection function
def betterCanMove(entity, mazeArray, direction):

    blocked = [1,3]
    horzMov = [LEFT, RIGHT]
    vertMov = [UP, DOWN]

    x = entity.x-block_size/2.0 #get the top left corner
    y = entity.y-block_size/2.0

    occupiedTiles = [[int(round((x)/block_size)),  int(round((y)/block_size))]] 

    for textX in range(0,2):
        for textY in range(0,2):
            testTileX = int(((x + textX * (block_size - 1))/block_size))
            testTileY = int(((y + textY * (block_size - 1))/block_size))
            testTile = [testTileX,testTileY]
            if(not testTile in occupiedTiles) and testTileX>=occupiedTiles[0][0]  and testTileY>=occupiedTiles[0][1]:
                occupiedTiles.append(testTile)

    #determine if its in motion 
    if len(occupiedTiles) == 2:
        if(occupiedTiles[0][0] == occupiedTiles[1][0] and direction in vertMov):
            return True
        elif(occupiedTiles[0][1] == occupiedTiles[1][1] and direction in horzMov):
            return True

    deltaX,deltaY = entity.COORD_DIR[direction]
     
    for tileX, tileY in occupiedTiles:
        #print("Testing: ({},{}){} Data: {}".format(tileX, tileY, direction, mazeArray[tileY+deltaY][tileX+deltaX]))

        if mazeArray[(tileY+deltaY)%len(mazeArray)][(tileX+deltaX)%len(mazeArray[0])] in blocked:
            return False

    return True
    
#human controlled pac_man
def humanPlayer(pac_man, maze, ghosts, pellets, power_pellets, fruit):
    # rotatingCameraNeatHelper(pac_man, maze, ghosts, pellets, power_pellets, fruit)

    x = pac_man.x-block_size/2.0 #get the top left corner
    y = pac_man.y-block_size/2.0

    occupiedTiles = [[int(round((x)/block_size)),  int(round((y)/block_size))]] 

    for textX in range(0,2):
        for textY in range(0,2):
            testTileX = int(((x + textX * (block_size - 1))/block_size))
            testTileY = int(((y+ textY * (block_size - 1))/block_size))
            testTile = [testTileX,testTileY]
            if(not testTile in occupiedTiles) and testTileX>=occupiedTiles[0][0]  and testTileY>=occupiedTiles[0][1]:
                occupiedTiles.append(testTile)
    

    #print(occupiedTiles)
    possible_dirs = []

    if(betterCanMove(pac_man, maze.maze_array, RIGHT)):
        possible_dirs.append(RIGHT)

    if(betterCanMove(pac_man, maze.maze_array, LEFT)):
        possible_dirs.append(LEFT)

    if(betterCanMove(pac_man, maze.maze_array, UP)):
        possible_dirs.append(UP)

    if(betterCanMove(pac_man, maze.maze_array, DOWN)):
        possible_dirs.append(DOWN)

    return pac_man.humanInput

#moves randomly but does not run into walls or ghosts 
def avoid_ghost_and_wall_dummy(pac_man, maze, ghosts, pellets, power_pellets, fruit):
    #evaluate walls 
    possible_dirs = []

    if(betterCanMove(pac_man, maze.maze_array, RIGHT)):
        possible_dirs.append(RIGHT)

    if(betterCanMove(pac_man, maze.maze_array, LEFT)):
        possible_dirs.append(LEFT)

    if(betterCanMove(pac_man, maze.maze_array, UP)):
        possible_dirs.append(UP)

    if(betterCanMove(pac_man, maze.maze_array, DOWN)):
        possible_dirs.append(DOWN)

    ghostTiles = []

    for ghost in ghosts.values():
        if ghost.mode == "normal" and ghost.blue == False:
            x = ghost.x-block_size/2.0 #get the top left corner
            y = ghost.y-block_size/2.0

            for textX in range(0,2):
                for textY in range(0,2):
                    testTile = [int(((x + textX * (block_size - 1))/block_size)),  int(((y+ textY * (block_size - 1))/block_size))]
                    if(not testTile in ghostTiles):
                        ghostTiles.append(testTile)

    pac_manTiles = []
    x = pac_man.x-block_size/2.0 #get the top left corner
    y = pac_man.y-block_size/2.0

    for textX in range(0,2):
        for textY in range(0,2):
            testTile = [int(((x + textX * (block_size - 1))/block_size)),  int(((y+ textY * (block_size - 1))/block_size))]
            if(not testTile in pac_manTiles):
                pac_manTiles.append(testTile)

    for direction in possible_dirs:
        testRange = 2
        deltaX,deltaY = pac_man.COORD_DIR[direction]

        tiles_to_test = []
        for tileX, tileY in pac_manTiles:
            for i in range(1, testRange+1):
                if(i==1 and deltaX == 0):
                    for cornercheck in range(-1,2):
                        testTile = [tileX+deltaX*i + cornercheck, tileY+deltaY*i]
                        if not testTile in tiles_to_test:
                            tiles_to_test.append(testTile)

                if(i==1 and deltaY == 0):
                    for cornercheck in range(-1,2):
                        testTile = [tileX+deltaX*i, tileY+deltaY*i+ cornercheck]
                        if not testTile in tiles_to_test:
                            tiles_to_test.append(testTile)

                else: 
                    testTile = [tileX+deltaX*i, tileY+deltaY*i]
                    if not testTile in tiles_to_test:
                        tiles_to_test.append(testTile)

        for testTile in tiles_to_test:
            if(testTile in ghostTiles):
                #print("AHHH ghost at ({},{}) in direction {}".format(testTile[0],testTile[1], direction))
                possible_dirs.remove(direction)
                break

    if len(possible_dirs) != 0 and (pac_man.oldPosDir != possible_dirs or not pac_man.move_dir in possible_dirs):
        pac_man.oldPosDir = copy.deepcopy(possible_dirs)
        return random.choice(possible_dirs)
    return pac_man.move_dir

# A queue node used in BFS -> taken and adapted from https://www.techiedelight.com/find-shortest-path-source-destination-matrix-satisfies-given-constraints/
class queueNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

def getPath(node, path):
    if node:
        getPath(node.parent, path)
        path.append(node)

def find_path_to_objective(start, objective, blocked, mazeArray,notAllowed = []):
    start

    start = queueNode(start[0], start[1])

    queue = list()
    visited = set()

    visited.add((start.x,start.y))

    queue.append(start)

    dirs = ((0, 1), (0, -1), (1, 0), (-1, 0))

    while queue:

        curr = queue.pop(0)
        
        if [curr.x, curr.y] in objective:
            path = []
            getPath(curr, path)
            return path

        #check adjacents and add to queue
        for testX, testY in dirs:
                x = (testX+curr.x)%len(mazeArray[0])
                y = (testY+curr.y)%len(mazeArray)

                
                if((not mazeArray[y][x] in blocked) and (not [x,y] in notAllowed) and (not (x,y) in notAllowed)):
                    nextCoord = queueNode(x,y, curr)
                    if (nextCoord.x,nextCoord.y) not in visited:
                        queue.append(nextCoord)
                        visited.add((nextCoord.x,nextCoord.y))

    return -1

#the final baseline level -> able to path find to a variety of targets such as pellets or blue ghosts 
def pathFind_to_target(pac_man, maze, ghosts, pellets, power_pellets, fruit):
    mazeArray = copy.deepcopy(maze.maze_array)

    #makes hostile ghosts appear as walls to the path finding algo
    for ghost in ghosts.values():
        if ghost.mode == "normal" and ghost.blue == False:
            x = ghost.x-block_size/2.0 #get the top left corner
            y = ghost.y-block_size/2.0

            for textX in range(0,2):
                for textY in range(0,2):
                    mazeArray[int(((y+ textY * (block_size - 1))/block_size))%len(mazeArray)][int(((x + textX * (block_size - 1))/block_size))%len(mazeArray[0])] = 1
               
    pac_manX = int(round((pac_man.x-block_size/2.0)/block_size))%len(mazeArray[0]) #get pacman location
    pac_manY = int(round((pac_man.y-block_size/2.0)/block_size))%len(mazeArray) 

    # Don't get power pellets until all ghosts are active
    if ghosts["clyde"].mode == "normal":
        blocked = [1,3]
    else:
        blocked = [1,3,4]

    path = -1
    timed_objectives = []

    #path finder 
    if pac_man.powered_up and pac_man.ghosts_eaten < 4:
        for ghost in ghosts.values():
            if ghost.mode == "normal" and ghost.blue:
                timed_objectives.append(ghost.array_coord)      
    
    if fruit.here:
        timed_objectives.append(fruit.array_coord)

    if timed_objectives:
        path = find_path_to_objective([pac_manX, pac_manY], timed_objectives, blocked, mazeArray)
    
    if path == -1:
        power_pellet_locs = []
        for power_pellet in power_pellets:
            if power_pellet.here:
                power_pellet_locs.append(power_pellet.array_coord)

        if power_pellet_locs and ghosts["clyde"].mode == "normal":
            path = find_path_to_objective([pac_manX, pac_manY], power_pellet_locs, blocked, mazeArray)

        if path == -1:
            pellet_locs = []
            for pellet in pellets:
                if pellet.here:
                    pellet_locs.append(pellet.array_coord)
            path = find_path_to_objective([pac_manX, pac_manY], pellet_locs, blocked, mazeArray)

    #if no path is avaliable panic
    if(path == -1):
        return avoid_ghost_and_wall_dummy(pac_man, maze, ghosts, pellets, power_pellets, fruit)
    elif len(path) < 2:
        return pac_man.move_dir
    else:
        nextTile = [path[1].x, path[1].y]
        if(nextTile[0]>pac_manX): return RIGHT
        if(nextTile[0]<pac_manX): return LEFT
        if(nextTile[1]>pac_manY): return DOWN
        if(nextTile[1]<pac_manY): return UP




