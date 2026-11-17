# Drawing the game as text, for watching it in the terminal (play.py --text)


def text_frame(env):
    rows = []
    for r in range(env.height):
        row = []
        for c in range(env.width):
            if env.walls[r, c]:
                row.append("%")
            else:
                row.append(" ")
        rows.append(row)

    for r, c in env.food:
        rows[r][c] = "."
    for r, c in env.capsules:
        rows[r][c] = "o"
    for i in range(len(env.ghosts)):
        r, c = env.ghosts[i]
        if env.scared[i] > 0:
            rows[r][c] = "g"
        else:
            rows[r][c] = "G"
    r, c = env.pacman
    rows[r][c] = "P"

    lines = []
    for row in rows:
        lines.append("".join(row))
    lines.append(f"score {env.score}   steps {env.steps}   food left {len(env.food)}")
    return "\n".join(lines)
