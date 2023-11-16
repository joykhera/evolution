import neat
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

class PlotReporter(neat.reporting.BaseReporter):
    def post_evaluate(self, config, population, species, best_genome):
        print('xxxp', config, population, species, best_genome)
        plt.ion()
        fig, ax = plt.subplots()
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best Fitness Over Generations")

        clear_output(wait=True)

        fitnesses = [c.fitness for c in stats.most_fit_genomes]
        ax.clear()
        ax.plot(fitnesses, label="Best Fitness")
        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.set_title("Best Fitness Over Generations")

        display(fig)
        plt.pause(0.001)
