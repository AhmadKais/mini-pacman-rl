# Live view while training: a chart that updates as the agent learns,
# and every few episodes the agent plays one game in a pygame window.
# Closing either window only stops the drawing, training keeps going.

import matplotlib.pyplot as plt

from pacman.render import PygameView


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
        self.view = PygameView(env, fps=fps)
        plt.ion()
        self.fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        self.score_ax = axes[0]
        self.win_ax = axes[1]
        self.fig.canvas.manager.set_window_title(name + " - learning curves")

    def update(self, env, agent, log):
        episode = len(log)
        self.view.handle_events()  # keeps the game window responding
        if episode % 10 == 0:
            self.redraw_chart(log, agent)
        if episode % self.watch_every == 0 and not self.view.closed:
            self.watch_game(env, agent, episode)

    def redraw_chart(self, log, agent):
        if not plt.fignum_exists(self.fig.number):
            return  # the chart window was closed
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
        self.score_ax.set_title("score")

        self.win_ax.clear()
        self.win_ax.plot(episodes, moving_average(wins), color="tab:green")
        self.win_ax.set_ylim(-5, 105)
        self.win_ax.set_title("win rate % (last 50 games)")

        for ax in [self.score_ax, self.win_ax]:
            ax.grid(alpha=0.3)
            ax.set_xlabel("episode")
        title = f"{self.name}   episode {episodes[-1]}   epsilon {agent.epsilon:.2f}"
        self.fig.suptitle(title)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

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
        plt.ioff()
        self.view.close()
        if plt.fignum_exists(self.fig.number):
            plt.show()  # leave the final chart open
