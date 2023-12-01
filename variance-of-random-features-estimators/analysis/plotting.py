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

def get_comparatives(column:str, df_with_n_components, df_labels:list, title:str, 
                     names_of_std_for_column:dict, 
                     constant_lines = [], constant_labels = [], constant_std = [], 
                     log_scale  = False, constant_margin = 0.001, marker = '',
                     percent_of_n_components_bigger_than = 0, 
                     column_x = "percent", axis_x_name = 'n_components in %'):

    alpha_std_area = 0.2
    # Set a larger figure size
    plt.figure(figsize=(16, 10))
    len_df = len(df_with_n_components)
    len_constant = len(constant_lines)
    # Plot dataframes
    dominium = []
    for df, label,color, name_std_column in zip(df_with_n_components, df_labels, colours[0:len_df], names_of_std_for_column):
        df = df[df[column_x]>= percent_of_n_components_bigger_than]
       
        plt.plot(df[column_x],  df[column], marker=marker,label= label, color = color)
        dominium = df[column_x]
        if name_std_column != False:
            std = df[name_std_column]
            # Crear el área sombreada alrededor de ±1 std
            plt.fill_between(df[column_x],df[column] - std , df[column] + std, alpha=alpha_std_area, color=color)

    # Plot n_components contantes
    for constant_value, label,std, color, name_std in zip(constant_lines, constant_labels,  constant_std,colours[len_df: len_df+len_constant],names_of_std_for_column ):
        plt.axhline(y=constant_value,  label=label, color=color, linestyle = "--")
        if name_std != False:
            plt.fill_between(dominium, constant_value - std, constant_value + std, alpha=alpha_std_area, color=color)
    
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
                      title:str, names_of_std_for_column:dict, 
                      constant_lines = [], constant_labels = [], constant_std = [], 
                      log_scale = False, constant_margin = 0.001, marker = '',
                      percent_of_n_components_bigger_than = 0,
                      column_x = "percent", axis_x_name = 'n_components in %'):
    get_comparatives(column, 
                     df_with_n_components, df_labels, title,names_of_std_for_column,
                       constant_lines , constant_labels,constant_std, log_scale, 
                       constant_margin, marker,
                       percent_of_n_components_bigger_than,
                       column_x = column_x, axis_x_name = axis_x_name)
    plt.show()


def view_plots_and_save_them(df_list:list, df_list_names:list, names_of_std_for_column:list, 
                            type:str,
                            columns:list, database:str, plot_path:str,
                            df_constant: list, constant_labels:list,
                            percent_of_n_components_bigger_than = 0, 
                            column_x = "percent", axis_x_name = 'n_components in %'):
    '''Muestra y guarda en `path` la lista con los gráficos en función de su número de componentes. Los argumentos son:
    `df_list`: Listas de dataframes que va a mostrarse, debe de tener una columna con `n-components`. Si no la tiene deberá de guardarse en `df_constant`. Debe contener las columnas especificadas en columns.
    `df_list_names`: Etiqueta que se le asignará en el gráfico a cada dataframe.
    `type`: título adicional que se añadirá al título. 
    `columns:list`: Lista de las columnas del df_list de las que se va a sacar una gráfica. 
    `database:str`: Información del dataset para el título
    `plot_path:str`: Ruta del directorio en la que se van a guardar los gráficos.
    `df_constant: list`: lista con los dataframes que van a ser constantantes. Debe contener las columnas 
    `constant_labels:list`: Etiqueta que se le asignará en el gráfico a cada dataframe. Es dependiente del orden.
    `percent_of_n_components_bigger_than` El número de componentes a partir del cual se muestra. Será los valores  de la columna `columna_x` que se muestran.
    `column_x = "percent"` columna de los valores de x sobre los que se muestrea el gráfico.
    `axis_x_name = 'n_components in %' `: Títlo x de la imagen que se guarda. 

    '''
    for column, std_column in zip(columns, names_of_std_for_column) :
        title = f"{type}{column} {database}"
        title += f' starting at {percent_of_n_components_bigger_than} percent'

        constant_lines = []
        constant_std = []
        for df in df_constant:
            if column in df.columns:
                constant_lines.append(df[column].to_list()[0])
                constant_std.append(df[std_column].to_list()[0])

        _names_of_std_for_column = [std_column for _ in range(len(df_list))]
        get_comparatives(column = column, df_with_n_components= df_list, 
                         df_labels=df_list_names, 
                         title = title, 
                         names_of_std_for_column = _names_of_std_for_column,
                        constant_lines= constant_lines, 
                        constant_labels=constant_labels, constant_std= constant_std,
                        percent_of_n_components_bigger_than=percent_of_n_components_bigger_than,
                        column_x = column_x, axis_x_name = axis_x_name)
        save_route =  plot_path+ title.replace(' ', '-')
        # Guarda el gráfico en la ubicación especificada por save_path
        plt.savefig(save_route, bbox_inches='tight')  # bbox_inches='tight' ajusta los márgenes para que se ajusten correctamente
        
        # Muestra el gráfico en pantalla
        plt.show()

