#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 11 - Usar varias capas de neuronas
#
# Módulos necesarios:
#   NUMPY 1.16.3
#   MATPLOTLIB: 3.0.3
#   TENSORFLOW: 1.13.1
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf=tf.compat.v1
tf.disable_eager_execution()
class Perceptron_multicapa:
    def __init__(self):
        self.valores_entradas_X = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
        self.valores_a_predecir_Y = [[0.], [1.], [1.], [0.]]


    def parametros(self):
        #-------------------------------------
        #    PARÁMETROS DE LA RED
        #-------------------------------------
        #Variable TensorFLow correspondiente a los valores de neuronas de entrada
        self.tf_neuronas_entradas_X = tf.placeholder(tf.float32, [None, 2])

        #Variable TensorFlow correspondiente a la neurona de salida (predicción real)
        self.tf_valores_reales_Y = tf.placeholder(tf.float32, [None, 1])

        #Cantidad de neuronas en la capa oculta
        self.n_neuronas_capa_oculta = 2

    def pesos(self):
        #PESOS
        #Los primeros están 4 : 2 en la entrada (X1 y X2) y 2 pesos por entrada
        self.pesos = tf.Variable(tf.random_normal([2, 2]), tf.float32)

        #los pesos de la capa oculta están 2 : 2 en la entrada (H1 y H2) y 1 peso por entrada
        self.peso_capa_oculta = tf.Variable(tf.random_normal([2, 1]), tf.float32)

        #El primer sesgo contiene 2 pesos
        self.sesgo = tf.Variable(tf.zeros([2]))

        #El segundo sesgo contiene 1 peso
        self.sesgo_capa_oculta = tf.Variable(tf.zeros([1]))

    def activación(self):
        #Cálculo de la activación de la primera capa
        #cálculo de la suma ponderada (tf.matmul) con ayuda de los datos X1, X2, W11, W12, W31, W41 y del sesgo
        #después aplicación de la función sigmoide (tf.sigmoid)
        activacion = tf.sigmoid(tf.matmul(self.tf_neuronas_entradas_X, self.pesos) + self.sesgo)

        #Cálculo de la activación de la capa oculta
        #cálculo de la suma ponderada (tf.matmul) con ayuda de los datos H1, H2, W12, W21 y del sesgo
        #después aplicación de la función sigmoide (tf.sigmoid)
        self.activacion_capa_oculta = tf.sigmoid(tf.matmul(activacion, self.peso_capa_oculta) + self.sesgo_capa_oculta)

        #Función de error de media cuadrática MSE
        self.funcion_error = tf.reduce_sum(tf.pow(self.tf_valores_reales_Y-self.activacion_capa_oculta,2))

        #Descenso del gradiente con una tasa de aprendizaje fijada en 0,1
        self.optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.funcion_error)

    def iniciacion(self):
        #Cantidad de epochs
        epochs = 1000

        #Inicialización de la variable
        init = tf.global_variables_initializer()

        #Inicio de una sesión de aprendizaje
        self.sesion = tf.Session()
        self.sesion.run(init)

        #Para la realización de la gráfica para la MSE
        self.Grafica_MSE=[]


        #Para cada epoch
        for i in range(epochs):

            #Realización del aprendizaje con actualización de los pesos
            self.sesion.run(self.optimizador, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y:self.valores_a_predecir_Y})

            #Calcular el error
            MSE = self.sesion.run(self.funcion_error, feed_dict = {self.tf_neuronas_entradas_X: self.valores_entradas_X, self.tf_valores_reales_Y:self.valores_a_predecir_Y})

            #Visualización de la información
            self.Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(epochs) + ") -  MSE: "+ str(MSE))

    def visualizacion(self):
    #Visualización gráfica
        plt.plot(self.Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()


        print("--- VERIFICACIONES ----")

        for i in range(0,4):
            print("Observación:"+str(self.valores_entradas_X[i])+ " - Esperado: "+str(self.valores_a_predecir_Y[i])+" - Predicción: "+str(self.sesion.run(self.activacion_capa_oculta, feed_dict={self.tf_neuronas_entradas_X: [self.valores_entradas_X[i]]})))



        self.sesion.close()

    @staticmethod
    def ejecutar():
        perceptron= Perceptron_multicapa()
        perceptron.parametros()
        perceptron.pesos()
        perceptron.activación()
        perceptron.iniciacion()
        perceptron.visualizacion()


if __name__=='__main__':

    Perceptron_multicapa.ejecutar()
