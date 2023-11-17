import neat
from IPython.display import display, clear_output
import matplotlib.pyplot as plt


class PlotReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.best_fitnesses = []
        self.average_fitnesses = []
        self.fig, self.ax = plt.subplots()

    def post_evaluate(self, config, population, species, best_genome):
        self.best_fitnesses.append(best_genome.fitness)
        fitnesses = []

        for genome in population.values():
            fitnesses.append(genome.fitness)

        average_fitness = sum(fitnesses) / len(fitnesses)
        self.average_fitnesses.append(average_fitness)

        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.title("Best Fitness Over Generations")

        clear_output(wait=True)
        self.ax.clear()
        self.ax.plot(self.best_fitnesses, label="Best Fitness")
        self.ax.plot(self.average_fitnesses, label="Average Fitness")
        self.ax.set_xlabel("Generations")
        self.ax.set_ylabel("Fitness")
        self.ax.set_title("Fitness Over Generations")
        self.ax.legend(loc="upper left")

        display(self.fig)
        plt.pause(0.001)
        self.fig.savefig("fitness.png")
