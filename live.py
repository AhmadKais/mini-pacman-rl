# Live view while training: a chart that updates as the agent learns,
# and every few episodes the current agent plays one game in a pygame window.

import matplotlib.pyplot as plt

from render import PygameView


def moving_avg(values, k=50):
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= k:
            total -= values[i - k]
        out.append(total / min(i + 1, k))
    return out


class LiveTraining:
    def __init__(self, env, name, watch_every=100, fps=15):
        self.name = name
        self.watch_every = watch_every
        self.view = PygameView(env, fps=fps)
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(11, 4))
        self.fig.canvas.manager.set_window_title(f"{name} - learning curves")

    def update(self, env, agent, log):
        episode = len(log)
        self.view.handle_events()  # keep the game window responsive
        if episode % 10 == 0:
            self.redraw_plot(log, agent)
        if episode % self.watch_every == 0 and not self.view.closed:
            self.watch_game(env, agent, episode)

    def redraw_plot(self, log, agent):
        if not plt.fignum_exists(self.fig.number):
            return  # chart window was closed, just keep training
        eps = [x[0] for x in log]
        scores = [x[1] for x in log]
        wins = [x[2] * 100 for x in log]
        for ax in (self.ax1, self.ax2):
            ax.clear()
            ax.grid(alpha=0.3)
            ax.set_xlabel("episode")
        self.ax1.plot(eps, scores, color="lightgray", linewidth=0.8)
        self.ax1.plot(eps, moving_avg(scores), color="tab:blue")
        self.ax1.set_title("score")
        self.ax2.plot(eps, moving_avg(wins), color="tab:green")
        self.ax2.set_ylim(-5, 105)
        self.ax2.set_title("win rate % (last 50 games)")
        self.fig.suptitle(f"{self.name}   episode {eps[-1]}   epsilon {agent.epsilon:.2f}")
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def watch_game(self, env, agent, episode):
        # play one game with no exploration so we see what it has learned so far
        self.view.pg.display.set_caption(f"{self.name} - after {episode} episodes")
        obs = env.reset()
        action = None
        self.view.draw(env, action)
        while not env.done and not self.view.closed:
            action = agent.act(env, obs, greedy=True)
            obs, _, _, _ = env.step(action)
            self.view.draw(env, action)
        if not self.view.closed:
            self.view.pg.time.wait(700)

    def close(self):
        plt.ioff()
        self.view.close()
        if plt.fignum_exists(self.fig.number):
            plt.show()  # leave the final chart open
