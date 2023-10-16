from sklearn.kernel_approximation import Nystroem

random_state = 123

def get_Nystroem_transformer(gamma:float, n_component:int, random_state = random_state):
    return Nystroem(gamma=gamma,  kernel='rbf',
             n_components=n_component, random_state=random_state)