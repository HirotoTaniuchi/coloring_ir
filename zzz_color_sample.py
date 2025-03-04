import matplotlib.pyplot as plt

def plot_color_map(color_map):
    fig, axes = plt.subplots(1, len(color_map), figsize=(len(color_map) * 2, 2))
    if len(color_map) == 1:
        axes = [axes]
    for idx, (key, color) in enumerate(color_map.items()):
        axes[idx].imshow([[color]], extent=(0, 100, 0, 100))
        axes[idx].set_title(f'Label {key}')
        axes[idx].axis('off')
    plt.savefig('colormap.png')



# Define the color map for values 0 to 8
color_map = {
    "0 Unlabelled": (0, 0, 0),       # Unlabelled?
    "1 Car" : (57, 5, 126),    # Car?
    "2 Person" : (64, 64, 16),    # Person?
    "3 Bike" : (67, 126, 186),  # Bike?
    "4 Curve": (32, 9, 181),    # Curve?
    "5 Car stop" : (128, 127, 39),  # Car Stop?
    "6 Guardrail": (63, 64, 124),   # Guardrail
    "7 Color cone": (183, 132, 128), # Color cone?
    "8 Bump": (177, 75, 30)    # Bump?
}

if __name__ == "__main__":
    plot_color_map(color_map)