# Drawing the game in a pygame window.
# Keys: up arrow = faster, down arrow = slower, space = pause.

import math

import pygame

CELL = 40
WALL_COLOR = (30, 60, 200)
FOOD_COLOR = (255, 230, 180)
PACMAN_COLOR = (255, 220, 0)
SCARED_COLOR = (40, 40, 255)
GHOST_COLORS = [(230, 50, 50), (255, 150, 200), (60, 220, 230), (255, 170, 60)]
MOUTH_ANGLES = [-90, 90, 180, 0]  # up, down, left, right


class PygameView:
    def __init__(self, env, fps=8, panel_width=0, min_height=0):
        # panel_width: extra space on the right (the live training chart goes there)
        pygame.init()
        self.fps = fps
        self.game_width = env.width * CELL
        height = max(env.height * CELL + 30, min_height)
        self.screen = pygame.display.set_mode((self.game_width + panel_width, height))
        pygame.display.set_caption("Mini Pac-Man RL")
        self.font = pygame.font.SysFont("consolas", 18)
        self.clock = pygame.time.Clock()
        self.closed = False
        self.paused = False

    def center(self, cell):
        x = cell[1] * CELL + CELL // 2
        y = cell[0] * CELL + CELL // 2
        return x, y

    def handle_events(self):
        # returns False once the user has closed the window
        if self.closed:
            return False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            if event.type == pygame.KEYDOWN:
                self.handle_key(event.key)
        return True

    def handle_key(self, key):
        if key == pygame.K_UP:
            self.fps = min(60, self.fps * 2)
        elif key == pygame.K_DOWN:
            self.fps = max(1, self.fps // 2)
        elif key == pygame.K_SPACE:
            self.paused = not self.paused

    def wait_while_paused(self):
        if self.paused:
            text = self.font.render("PAUSED (space)", True, (255, 255, 255))
            self.screen.blit(text, (8, 5))
            pygame.display.flip()
        while self.paused and self.handle_events():
            self.clock.tick(20)

    def draw(self, env, action=None):
        if not self.handle_events():
            return
        self.screen.fill((0, 0, 0), (0, 0, self.game_width, self.screen.get_height()))
        for r in range(env.height):
            for c in range(env.width):
                if env.walls[r, c]:
                    rect = (c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4)
                    pygame.draw.rect(self.screen, WALL_COLOR, rect, border_radius=6)
        for cell in env.food:
            pygame.draw.circle(self.screen, FOOD_COLOR, self.center(cell), CELL // 10)
        for cell in env.capsules:
            pygame.draw.circle(self.screen, FOOD_COLOR, self.center(cell), CELL // 4)
        for i in range(len(env.ghosts)):
            if env.scared[i] > 0:
                color = SCARED_COLOR
            else:
                color = GHOST_COLORS[i % len(GHOST_COLORS)]
            self.draw_ghost(env.ghosts[i], color)
        self.draw_pacman(env.pacman, action)
        self.draw_status(env)
        pygame.display.flip()
        self.clock.tick(self.fps)
        self.wait_while_paused()

    def draw_ghost(self, cell, color):
        x, y = self.center(cell)
        radius = CELL // 2 - 4
        pygame.draw.circle(self.screen, color, (x, y - 2), radius)
        pygame.draw.rect(self.screen, color, (x - radius, y - 2, 2 * radius, radius))
        pygame.draw.circle(self.screen, (255, 255, 255), (x - radius // 2, y - 5), 4)
        pygame.draw.circle(self.screen, (255, 255, 255), (x + radius // 2, y - 5), 4)

    def draw_pacman(self, cell, action):
        x, y = self.center(cell)
        radius = CELL // 2 - 4
        pygame.draw.circle(self.screen, PACMAN_COLOR, (x, y), radius)
        # the mouth is a black triangle pointing where he last moved
        angle = 0
        if action is not None:
            angle = MOUTH_ANGLES[action]
        mouth = [(x, y)]
        for offset in [-35, 35]:
            a = math.radians(angle + offset)
            mouth.append((x + radius * 1.2 * math.cos(a), y + radius * 1.2 * math.sin(a)))
        pygame.draw.polygon(self.screen, (0, 0, 0), mouth)

    def draw_status(self, env):
        msg = f"score {env.score}  steps {env.steps}  food {len(env.food)}"
        if env.won:
            msg += "   WIN!"
        elif env.truncated:
            msg += "   time up"
        elif env.done:
            msg += "   CAUGHT"
        text = self.font.render(msg, True, (255, 255, 255))
        self.screen.blit(text, (8, env.height * CELL + 5))

    def set_title(self, title):
        if not self.closed:
            pygame.display.set_caption(title)

    def pause(self, milliseconds):
        if not self.closed:
            pygame.time.wait(milliseconds)

    def close(self):
        if not self.closed:
            self.closed = True
            pygame.quit()
