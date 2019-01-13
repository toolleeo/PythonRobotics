"""

A* grid based planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
import math
import sys

show_animation = True
pause_animation = False

def press(event):
    global pause_animation
    if event.key == 'p' or event.key == 'P':
        if pause_animation == True:
            pause_animation = False
        else:
            pause_animation = True
    if event.key == 'q' or event.key == 'Q':
        print('Quitting upon request.')
        sys.exit(0)

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Node(Point):
    def __init__(self, x, y, cost, pind):
        super(Node, self).__init__(x, y)
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)


class GridMap(object):
    def __init__(self, ox, oy, reso, robot_radius):
        self.ox = ox
        self.oy = oy
        self.reso = reso
        self.robot_radius = robot_radius

        self.ox = [iox / reso for iox in ox]
        self.oy = [ioy / reso for ioy in oy]

        self.obmap, self.minx, self.miny, self.maxx, self.maxy, self.xw, self.yw = calc_obstacle_map(self.ox, self.oy, self.reso, self.robot_radius)
        print('xw {} yw {}'.format(self.xw, self.yw))

    def verify_node(self, node):

        if node.x < self.minx:
            return False
        elif node.y < self.miny:
            return False
        elif node.x >= self.maxx:
            return False
        elif node.y >= self.maxy:
            return False

        if self.obmap[node.x][node.y]:
            return False

        return True

    def calc_final_path(self, nodes, ngoal):
        """ Generate final path."""

        rx, ry = [ngoal.x * self.reso], [ngoal.y * self.reso]
        pind = ngoal.pind
        while pind != -1:
            n = nodes[pind]
            rx.append(n.x * self.reso)
            ry.append(n.y * self.reso)
            pind = n.pind
        return rx, ry

    def image(self):
        max_col = round(self.xw * self.reso)
        max_row = round(self.yw * self.reso)
        image = [[0 for i in range(max_col)] for i in range(max_row)]
        #print('max: ({}, {})'.format(max_row, max_col))
        for row in range(max_row):
            for col in range(max_col):
                #print('row, col = ({}, {})'.format(row, col))
                row_i = math.floor(row / self.reso)
                col_i = math.floor(col / self.reso)
                #print('row_i, col_i = ({}, {})'.format(row_i, col_i))
                if self.obmap[row_i][col_i] == 1:
                    image[col][row] = 1
        return image



class AstarPlanner(object):
    def __init__(self, obmap, start, goal):
        """
        start: Point of the start position [m]
        goal: Point of the goal position [m]
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        reso: grid resolution [m]
        robot_radius: robot radius [m]
        """

        self.start = start
        self.goal = goal
        self.obmap = obmap

        self.motion = get_motion_model()

        self.nstart = Node(round(start.x / obmap.reso), round(start.y / obmap.reso), 0.0, -1)
        self.ngoal = Node(round(goal.x / obmap.reso), round(goal.y / obmap.reso), 0.0, -1)

        self.openset, self.closedset = dict(), dict()
        self.openset[calc_index(self.nstart, self.obmap.xw, self.obmap.minx, self.obmap.miny)] = self.nstart

        self.terminated = False
        self.found = False

    def calc_heuristic(self, n1, n2):
        w = 1.0  # weight of heuristic
        d = w * math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
        return d

    def planning_step(self):
        if len(self.openset) == 0:
            self.terminated = True
            self.found = False
            return

        c_id = min(
            self.openset, key=lambda o: self.openset[o].cost + self.calc_heuristic(self.ngoal, self.openset[o]))
        self.current = self.openset[c_id]

        if self.current.x == self.ngoal.x and self.current.y == self.ngoal.y:
            self.ngoal.pind = self.current.pind
            self.ngoal.cost = self.current.cost
            self.terminated = True
            self.found = True
            return

        # Remove the item from the open set
        del self.openset[c_id]
        # Add it to the closed set
        self.closedset[c_id] = self.current

        # Expand search grid based on motion model
        for i in range(len(self.motion)):
            node = Node(self.current.x + self.motion[i][0],
                        self.current.y + self.motion[i][1],
                        self.current.cost + self.motion[i][2], c_id)
            n_id = calc_index(node, self.obmap.xw, self.obmap.minx, self.obmap.miny)

            if n_id in self.closedset:
                continue

            if not self.obmap.verify_node(node):
                continue

            if n_id not in self.openset:
                self.openset[n_id] = node  # Discover a new node
            else:
                if self.openset[n_id].cost >= node.cost:
                    # This path is the best until now. record it!
                    self.openset[n_id] = node


def calc_obstacle_map(ox, oy, reso, robot_radius):
    """Calculates the occupancy grid based on positions of obstacles.

    Generates a grid of squared cells that are marked True or False
    depending on whether the cell contains an obstacle or not.

    ox: list of x co-ordinates of obstacles [m]
    oy: list of y co-ordinates of obstacles [m]
    reso: size of the grid cell [m]
    robot_radius: radius of the moving robot [m]

    Output:
    obmap: matrix of boolean flags
    minx: minimum x co-ordinate [m]
    miny: minimum y co-ordinate [m]
    maxx: maximum x co-ordinate [m]
    maxy: maximum y co-ordinate [m]
    xwidth: width of the grid map [m]
    ywidth: height of the grid map [m]
    """

    minx = math.floor(min(ox))
    miny = math.floor(min(oy))
    maxx = math.ceil(max(ox))
    maxy = math.ceil(max(oy))
    #  print("minx:", minx)
    #  print("miny:", miny)
    #  print("maxx:", maxx)
    #  print("maxy:", maxy)

    xwidth = maxx - minx
    ywidth = maxy - miny
    #  print("xwidth:", xwidth)
    #  print("ywidth:", ywidth)

    # obstacle map generation
    obmap = [[0 for i in range(xwidth)] for i in range(ywidth)]
    for ix in range(xwidth):
        x = ix + minx
        for iy in range(ywidth):
            y = iy + miny
            #  print(x, y)
            for iox, ioy in zip(ox, oy):
                d = math.sqrt((iox - x)**2 + (ioy - y)**2)
                if d <= robot_radius / reso:
                    obmap[ix][iy] = 1
                    break
    return obmap, minx, miny, maxx, maxy, xwidth, ywidth


def calc_index(node, xwidth, xmin, ymin):
    return (node.y - ymin) * xwidth + (node.x - xmin)


def get_motion_model():
    # dx, dy, cost
    motion = [[1, 0, 1],
              [0, 1, 1],
              [-1, 0, 1],
              [0, -1, 1],
              [-1, -1, math.sqrt(2)],
              [-1, 1, math.sqrt(2)],
              [1, -1, math.sqrt(2)],
              [1, 1, math.sqrt(2)]]

    return motion


def generate_obstacles():
    ox, oy = [], []
    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)
    return ox, oy


def main():
    global pause_animation
    print(__file__ + " started.")

    # start and goal position
    start = Point(10.0, 10.0)  # [m]
    goal = Point(50.0, 50.0)  # [m]
    grid_size = 2.0  # [m]
    robot_size = 1.0  # [m]

    ox, oy = generate_obstacles()

    obmap = GridMap(ox, oy, grid_size, robot_size)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(start.x, start.y, "xr")
        plt.plot(goal.x, goal.y, "xb")
        plt.axis("equal")

    planner = AstarPlanner(obmap, start, goal)

    colors = ['white', 'yellow', 'red', 'pink', 'black', 'green', 'orange']
    levels = [0, 1, 2, 3, 4, 5, 6, 7]
    cmap, norm = from_levels_and_colors(levels, colors)

    if show_animation:
        major_xticks = [x * grid_size for x in range(obmap.minx, obmap.maxx + 1)]
        major_yticks = [y * grid_size for y in range(obmap.miny, obmap.maxy + 1)]
        plt.xticks(major_xticks)
        plt.yticks(major_yticks)
        plt.imshow(obmap.image(), cmap=cmap, norm=norm)

    while planner.terminated == False:
        if pause_animation:
            plt.pause(0.1)
            continue
        # perform a planning step and prints the map
        planner.planning_step()
        # show graph
        if show_animation:
            plt.plot(planner.current.x * obmap.reso, planner.current.y * obmap.reso, "xc")
            plt.grid(True)
            if len(planner.closedset.keys()) % 10 == 0:
                plt.pause(0.001)

    rx, ry = obmap.calc_final_path(planner.closedset, planner.ngoal)

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.show()


if __name__ == '__main__':
    main()
