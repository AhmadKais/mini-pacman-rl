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
}


def load_layout(name):
    text = LAYOUTS.get(name)
    if text is None:
        # allow passing a path to a .txt file with a custom maze
        with open(name) as f:
            text = f.read()
    rows = [line for line in text.splitlines() if line.strip()]

    h, w = len(rows), max(len(r) for r in rows)
    walls = np.ones((h, w), dtype=bool)
    food, capsules, ghosts = set(), set(), []
    pacman = None

    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
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
        raise ValueError("layout has no pacman (P)")
    return walls, food, capsules, pacman, ghosts


def maze_distances(walls):
    # BFS from every open cell -> dist[a][b] = shortest path length from a to b
    dist = {}
    open_cells = [(int(r), int(c)) for r, c in zip(*np.where(~walls))]
    for start in open_cells:
        seen = {start: 0}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for n in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
                if not walls[n] and n not in seen:
                    seen[n] = seen[(r, c)] + 1
                    queue.append(n)
        dist[start] = seen
    return dist
