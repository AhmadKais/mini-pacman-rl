# Live view while training: one pygame window with the game on the left and the
# learning curves on the right. Every few episodes the agent plays one game there.
# The chart is drawn by matplotlib into an image and pygame shows it. (A separate
# matplotlib window and a pygame window in the same program can crash on Windows.)
# Closing the window only stops the drawing, training keeps going.

import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from pacman.render import PygameView

CHART_WIDTH = 480
CHART_HEIGHT = 420


def moving_average(values, window=50):
    result = []
    total = 0.0
    for i in range(len(values)):
        total += values[i]
        if i >= window:
            total -= values[i - window]
        count = min(i + 1, window)
        result.append(total / count)
    return result


class LiveTraining:
    def __init__(self, env, name, watch_every=100, fps=15):
        self.name = name
        self.watch_every = watch_every
        self.view = PygameView(env, fps=fps, panel_width=CHART_WIDTH, min_height=CHART_HEIGHT)
        self.view.set_title(name)
        height = self.view.screen.get_height()
        self.fig = Figure(figsize=(CHART_WIDTH / 100, height / 100), dpi=100)
        self.score_ax = self.fig.add_subplot(2, 1, 1)
        self.win_ax = self.fig.add_subplot(2, 1, 2)

    def update(self, env, agent, log):
        episode = len(log)
        self.view.handle_events()  # keeps the window responding
        if episode % 10 == 0:
            self.redraw_chart(log, agent)
        if episode % self.watch_every == 0 and not self.view.closed:
            self.watch_game(env, agent, episode)

    def redraw_chart(self, log, agent):
        if self.view.closed:
            return
        episodes = []
        scores = []
        wins = []
        for episode, score, won in log:
            episodes.append(episode)
            scores.append(score)
            wins.append(won * 100)

        self.score_ax.clear()
        self.score_ax.plot(episodes, scores, color="lightgray", linewidth=0.8)
        self.score_ax.plot(episodes, moving_average(scores), color="tab:blue")
        self.score_ax.set_title("score", fontsize=10)

        self.win_ax.clear()
        self.win_ax.plot(episodes, moving_average(wins), color="tab:green")
        self.win_ax.set_ylim(-5, 105)
        self.win_ax.set_title("win rate % (last 50 games)", fontsize=10)
        self.win_ax.set_xlabel("episode")

        for ax in [self.score_ax, self.win_ax]:
            ax.grid(alpha=0.3)
        self.fig.suptitle(f"episode {episodes[-1]}   epsilon {agent.epsilon:.2f}", fontsize=11)
        self.fig.tight_layout()
        self.show_chart()

    def show_chart(self):
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        size = canvas.get_width_height()
        image = pygame.image.frombuffer(canvas.buffer_rgba(), size, "RGBA")
        self.view.screen.blit(image, (self.view.game_width, 0))
        pygame.display.flip()

    def watch_game(self, env, agent, episode):
        # one game without random moves, to see what it has learned so far
        self.view.set_title(f"{self.name} - after {episode} episodes")
        obs = env.reset()
        action = None
        self.view.draw(env, action)
        while not env.done and not self.view.closed:
            action = agent.act(env, obs, greedy=True)
            obs, reward, done, info = env.step(action)
            self.view.draw(env, action)
        self.view.pause(700)

    def close(self):
        # leave the final chart open until the window is closed
        self.view.set_title(self.name + " - finished, close the window to exit")
        while self.view.handle_events():
            self.view.clock.tick(20)
