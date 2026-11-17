# Maze layouts for mini pacman
# % = wall, . = food, o = capsule, P = pacman, G = ghost

from collections import deque

import numpy as np

LAYOUTS = {
    "tiny": """
%%%%%%%
%P...o%
%.%%%.%
%....G%
%%%%%%%
""",
    "small": """
%%%%%%%%%
%o.....G%
%.%%.%%.%
%.......%
%.%%.%%.%
%P.....o%
%%%%%%%%%
""",
    "medium": """
%%%%%%%%%%%%%
%o....%....o%
%.%%%.%.%%%.%
%...........%
%.%%.%%%.%%.%
%....G.G....%
%.%%.%%%.%%.%
%.....P.....%
%%%%%%%%%%%%%
""",
    # never used for training, only to test how well the agents generalize
    "unseen": """
%%%%%%%%%%%
%o...%...o%
%.%%.%.%%.%
%.........%
%%.%%.%%.%%
%...G.G...%
%.%%.%.%%.%
%o...P...o%
%%%%%%%%%%%
""",
}


def read_layout(name):
    if name in LAYOUTS:
        text = LAYOUTS[name]
    elif "\n" in name:
        text = name  # the maze itself was passed in (used by the tests)
    else:
        # you can also give the path of a text file with your own maze
        with open(name) as f:
            text = f.read()

    rows = []
    for line in text.splitlines():
        if line.strip() != "":
            rows.append(line)

    height = len(rows)
    width = 0
    for line in rows:
        width = max(width, len(line))

    walls = np.ones((height, width), dtype=bool)
    food = set()
    capsules = set()
    ghosts = []
    pacman = None

    for r in range(height):
        for c in range(len(rows[r])):
            ch = rows[r][c]
            if ch == "%":
                continue
            walls[r, c] = False
            if ch == ".":
                food.add((r, c))
            elif ch == "o":
                capsules.add((r, c))
            elif ch == "P":
                pacman = (r, c)
            elif ch == "G":
                ghosts.append((r, c))

    if pacman is None:
        raise ValueError("the layout has no pacman (P)")
    return walls, food, capsules, pacman, ghosts


def maze_distances(walls):
    # dist[a][b] = number of steps to walk from cell a to cell b
    height, width = walls.shape
    dist = {}
    for r in range(height):
        for c in range(width):
            if not walls[r, c]:
                dist[(r, c)] = bfs(walls, (r, c))
    return dist


def bfs(walls, start):
    steps = {start: 0}
    queue = deque()
    queue.append(start)
    while len(queue) > 0:
        r, c = queue.popleft()
        neighbours = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        for cell in neighbours:
            if walls[cell] or cell in steps:
                continue
            steps[cell] = steps[(r, c)] + 1
            queue.append(cell)
    return steps
