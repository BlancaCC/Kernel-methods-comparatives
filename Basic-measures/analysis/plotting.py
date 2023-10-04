import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import itertools

colours = list(mcolors.CSS4_COLORS.keys())
def color_vector_features(color, luminiscente_coefficients = 0.6):

    red, green, blue = mcolors.to_rgb(color)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue

    adjust = lambda x: (1- luminiscente_coefficients )* x  + luminiscente_coefficients* luminance
    features = [
        adjust(red - green - blue),
        adjust(green - blue - red),
        adjust(blue - red - green),
        adjust(red + green - blue),
        adjust(red + blue - green),
        adjust(green + blue - red),
        adjust(green + blue + red),
    ]
    return features

mapping = [ (c,*color_vector_features(c)) for c in colours]
n_features = len(mapping[1])

get_index = lambda i: lambda v: (v[0], v[i])

def ordenar_y_extraer_nombres(indice):
    lista = list(map( get_index(indice), mapping ))
    lista_ordenada = sorted(lista, key=lambda x: x[1], reverse= False)
    nombres = [nombre for nombre, _ in lista_ordenada]
    return nombres

colours = list(itertools.chain.from_iterable(list(zip(*[ ordenar_y_extraer_nombres( i) for i in range(1,n_features)]))))

colours = colours[3:]

def get_comparatives(column:str, df_with_n_components, df_labels:list, title:str, std_column:dict, 
                     constant_lines = [], constant_labels = [], constant_std = [], log_scale  = False, constant_margin = 0.001, marker = ''):
    axis_x_name = 'n_components in %'
    column_x = "percent"
    alpha_std_area = 0.2
    # Set a larger figure size
    plt.figure(figsize=(16, 10))
    len_df = len(df_with_n_components)
    len_constant = len(constant_lines)
    # Plot dataframes
    for df, label,color in zip(df_with_n_components, df_labels, colours[0:len_df]):
        plt.plot(df[column_x],  df[column], marker=marker,label= label, color = color)
        if std_column[column] != False:
            # Crear el área sombreada alrededor de ±1 std
            std = df[std_column[column]]
            print('Column ', column,'\n',
            std
            )
            plt.fill_between(df[column_x],(df[column] - std).to_list() , df[column] + std, alpha=alpha_std_area, color=color)


    # Plot n_components contantes
    for constant_value, label,std, color in zip(constant_lines, constant_labels,  constant_std,colours[len_df: len_df+len_constant]):
        plt.axhline(y=constant_value,  label=label, color=color, linestyle = "--")
        if std_column[column] != False:
            plt.fill_between(df[column_x], constant_value - std, constant_value + std, alpha=alpha_std_area, color=color)
    
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


def plot_comparatives(column:str, df_with_n_components, df_labels:list, 
                      title:str, std_column:dict, 
                      constant_lines = [], constant_labels = [], constant_std = [], 
                      log_scale = False, constant_margin = 0.001, marker = ''):
    get_comparatives(column, df_with_n_components, df_labels, std_column, title, constant_lines , constant_labels,constant_std, log_scale, constant_margin, marker)
    plt.show()

def view_plots_and_save_them(df_list:list, df_list_names:list, std_column:dict, type:str,
                          columns:list, database:str, plot_path:str):
    for column in columns:
        title = f"{type}_{column.replace(' ', '_')}_{database}"

        constant_lines = df_list[0][column].to_list()
        if std_column[column] == False:
            constant_std = [0]
        else:
            constant_std = df_list[0][std_column[column]]
        constant_labels = [df_list_names[0]]
        plt.clf()
        get_comparatives(column, df_list[1:], df_list_names[1:], title, std_column, constant_lines, constant_labels, constant_std)
        save_route =  plot_path+ title
        # Guarda el gráfico en la ubicación especificada por save_path
        plt.savefig(save_route, bbox_inches='tight')  # bbox_inches='tight' ajusta los márgenes para que se ajusten correctamente
        plt.clf()
        #plot_comparatives(column, df_list[1:], df_list_names[1:], title, std_column, constant_lines, constant_labels, constant_std)
        # Muestra el gráfico en pantalla
        #plt.show()

