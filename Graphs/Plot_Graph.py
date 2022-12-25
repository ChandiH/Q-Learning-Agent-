import numpy as np
import matplotlib.pyplot as plt


class Draw:
    @staticmethod
    def plot_line_graph(x, xTitle, name):
        # Create a figure with a larger size
        plt.figure(figsize=(10, 6))

        # Create a subplot for the first plot
        plt.plot(np.array(range(0, len(x))), x)
        plt.title(f"{xTitle}")

        # Save the plot as a high-resolution image file
        plt.savefig(f"{name}.png", dpi=300)

        # Show the figure
        plt.show()

    @staticmethod
    def subplot_line_graph(x, xTitle, y, yTitle, name):
        # Create a figure with a larger size
        plt.figure(figsize=(10, 6))

        # Create a subplot for the first plot
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(np.array(range(100)), x)
        ax1.set_title(f"{xTitle}")

        # Create a subplot for the second plot
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(np.array(range(100)), y)
        ax2.set_title(f"{yTitle}")

        # Save the plot as a high-resolution image file
        plt.savefig(f"{name}.png", dpi=300)

        # Show the figure
        plt.show()

    @staticmethod
    def plot_multiple_line_graph(x, xTitle, name):
        # Create a figure with a larger size
        plt.figure(figsize=(10, 6))
        for _ in x:
            plt.plot(_)
        plt.title(f"{xTitle}")
        # Save the plot as a high-resolution image file
        plt.savefig(f"{name}.png", dpi=300)
        # Show the figure
        plt.show()

    @staticmethod
    def subplot_multiple_line_graph(x, xTitle, y, yTitle, name):
        # Create a figure with a larger size
        plt.figure(figsize=(10, 6))

        # Create a subplot for the first plot
        ax1 = plt.subplot(2, 1, 1)
        for _ in x:
            ax1.plot(np.array(range(0, 100)), x)
        ax1.set_title(f"{xTitle}")

        # Create a subplot for the second plot
        ax2 = plt.subplot(2, 1, 2)
        for _ in y:
            ax2.plot(np.array(range(0, 100)), y)
        ax2.set_title(f"{yTitle}")

        # Save the plot as a high-resolution image file
        plt.savefig(f"{name}.png", dpi=300)

        # Show the figure
        plt.show()
