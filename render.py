# Drawing the game: plain text in the terminal, or a pygame window.

import math


def text_frame(env):
    rows = [["%" if env.walls[r, c] else " " for c in range(env.width)]
            for r in range(env.height)]
    for r, c in env.food:
        rows[r][c] = "."
    for r, c in env.capsules:
        rows[r][c] = "o"
    for (r, c), s in zip(env.ghosts, env.scared):
        rows[r][c] = "g" if s else "G"
    r, c = env.pacman
    rows[r][c] = "P"
    lines = ["".join(row) for row in rows]
    lines.append(f"score {env.score}   steps {env.steps}   food left {len(env.food)}")
    return "\n".join(lines)


class PygameView:
    def __init__(self, env, cell=40, fps=8):
        import pygame
        self.pg = pygame
        pygame.init()
        self.cell = cell
        self.fps = fps
        self.screen = pygame.display.set_mode((env.width * cell, env.height * cell + 30))
        pygame.display.set_caption("Mini Pac-Man RL")
        self.font = pygame.font.SysFont("consolas", 18)
        self.clock = pygame.time.Clock()
        self.closed = False
        self.ghost_colors = [(230, 50, 50), (255, 150, 200), (60, 220, 230), (255, 170, 60)]

    def center(self, pos):
        r, c = pos
        return (c * self.cell + self.cell // 2, r * self.cell + self.cell // 2)

    def draw(self, env, action=None):
        pg, cell = self.pg, self.cell
        if self.closed or not self.handle_events():
            return
        self.screen.fill((0, 0, 0))

        for r in range(env.height):
            for c in range(env.width):
                if env.walls[r, c]:
                    rect = (c * cell + 2, r * cell + 2, cell - 4, cell - 4)
                    pg.draw.rect(self.screen, (30, 60, 200), rect, border_radius=6)
        for p in env.food:
            pg.draw.circle(self.screen, (255, 230, 180), self.center(p), cell // 10)
        for p in env.capsules:
            pg.draw.circle(self.screen, (255, 230, 180), self.center(p), cell // 4)

        for i, (p, s) in enumerate(zip(env.ghosts, env.scared)):
            color = (40, 40, 255) if s else self.ghost_colors[i % 4]
            x, y = self.center(p)
            rad = cell // 2 - 4
            pg.draw.circle(self.screen, color, (x, y - 2), rad)
            pg.draw.rect(self.screen, color, (x - rad, y - 2, 2 * rad, rad))
            for dx in (-rad // 2, rad // 2):
                pg.draw.circle(self.screen, (255, 255, 255), (x + dx, y - 5), 4)

        # pacman: a yellow circle with a mouth facing the last move
        x, y = self.center(env.pacman)
        rad = cell // 2 - 4
        pg.draw.circle(self.screen, (255, 220, 0), (x, y), rad)
        angle = {0: -90, 1: 90, 2: 180, 3: 0}.get(action, 0)
        mouth = [(x, y)]
        for da in (-35, 35):
            a = math.radians(angle + da)
            mouth.append((x + rad * 1.2 * math.cos(a), y + rad * 1.2 * math.sin(a)))
        pg.draw.polygon(self.screen, (0, 0, 0), mouth)

        msg = f"score {env.score}  steps {env.steps}  food {len(env.food)}"
        if env.done:
            msg += "   WIN!" if env.won else ("   time up" if env.truncated else "   CAUGHT")
        txt = self.font.render(msg, True, (255, 255, 255))
        self.screen.blit(txt, (8, env.height * cell + 5))
        pg.display.flip()
        self.clock.tick(self.fps)

    def handle_events(self):
        # returns False once the user closes the window
        if self.closed:
            return False
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                self.close()
                return False
        return True

    def close(self):
        if not self.closed:
            self.closed = True
            self.pg.quit()
