from codigo.perceptron_multicapa import Perceptron_multicapa
from codigo.Mina_o_Roca_N_neuronas import Neuronas

if __name__=='__main__':
    print('¿Qué archivo quiere ejecutar?')
    eleccion=input('Perceptron multicapa [1] ó el programa de Neuronas [2]: ')

    if eleccion=='1':
        Perceptron_multicapa.ejecutar()

    else:
        Neuronas.ejecutar()
