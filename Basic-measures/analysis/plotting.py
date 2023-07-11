import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


colours = list(mcolors.CSS4_COLORS.keys())

def get_luminance(color):
    rgb = mcolors.to_rgb(color)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return luminance
# Calculate the luminance for each color
luminances = {color: get_luminance(color) for color in colours}

# Sort the colors by luminance in descending order
sorted_colors = sorted(luminances, key=luminances.get, reverse=False)

colours = [color for i, color in enumerate(sorted_colors) if i % 3 == 0]
def get_comparatives(column:str, df_with_n_components, df_labels:list, title:str, constant_lines = [], constants_labels = [], log_scale  = False, constant_margin = 0.001, marker = ''):
    axis_x_name = 'n_components'
    # Set a larger figure size
    plt.figure(figsize=(16, 10))
    len_df = len(df_with_n_components)
    len_constant = len(constant_lines)
    # Plot dataframes
    for df, label,color in zip(df_with_n_components,df_labels,colours[0:len_df] ):
        plt.plot(df['n_components'],  df[column], marker=marker,label= label, color = color)

    # Plot n_components contantes
    for constant_value, label, color in zip(constant_lines, constants_labels, colours[len_df: len_df+len_constant]):
        plt.axhline(y=constant_value,  label=label, color=color)
    if len_df == 0:
        y_min = min(constant_lines) *(1- constant_margin)
        y_max = max(constant_lines)* (1+constant_margin)
        plt.ylim(y_min, y_max)
    # Set the labels and title
    plt.xlabel(axis_x_name)
    plt.ylabel(column)
    plt.title(title)

    if log_scale:
        # Set y-axis scale to logarithmic
        plt.yscale('log')

    # Add a legend
    plt.legend()


def plot_comparatives(column:str, df_with_n_components, df_labels:list, title:str, constant_lines = [], constans_labels = [], log_scale = False, constant_margin = 0.001, marker = ''):
    get_comparatives(column, df_with_n_components, df_labels, title, constant_lines , constans_labels, log_scale, constant_margin, marker)
    plt.show()
