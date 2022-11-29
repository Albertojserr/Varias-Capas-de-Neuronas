#-----------------------------------------------------------------------------------------
# @Autor: Aurélien Vannieuwenhuyze
# @Empresa: Junior Makers Place
# @Libro:
# @Capítulo: 11 - Usar varias capas de neuronas
#
# Módulos necesarios:
#   PANDAS 0.24.2
#   NUMPY  1.16.3
#   MATPLOTLIB  3.0.3
#   TENSORFLOW  1.13.1
#
# Para instalar un módulo:
#   Haga clic en el menú File > Settings > Project:nombre_del_proyecto > Project interpreter > botón +
#   Introduzca el nombre del módulo en la zona de búsqueda situada en la parte superior izquierda
#   Elegir la versión en la parte inferior derecha
#   Haga clic en el botón install situado en la parte inferior izquierda
#-----------------------------------------------------------------------------------------

#---------------------------------------------
# CARGA DE OBSERVACIONES
#---------------------------------------------

import pandas as pnd
from sklearn import preprocessing
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt

tf=tf.compat.v1
tf.disable_eager_execution()

def red_neuronas_multicapa(observaciones_en_entradas, pesos, peso_sesgo):

    #Cálculo de la activación de la primera capa
    primera_activacion = tf.sigmoid(tf.matmul(observaciones_en_entradas, pesos['capa_entrada_hacia_oculta']) + peso_sesgo['peso_sesgo_capa_entrada_hacia_oculta'])

    #Cálculo de la activación de la segunda capa
    activacion_capa_oculta = tf.sigmoid(tf.matmul(primera_activacion, pesos['capa_oculta_hacia_salida']) + peso_sesgo['peso_sesgo_capa_oculta_hacia_salida'])

    return activacion_capa_oculta


class Neuronas:
    def __init__(self,N,epochs = 300,cantidad_neuronas_entrada = 60, cantidad_neuronas_salida = 2,tasa_aprendizaje = 0.01):
        self.observaciones = pnd.read_csv("datas/sonar.all-data.csv")
        self.numeroNeuronas=N
        self.epochs = epochs
        self.cantidad_neuronas_entrada = cantidad_neuronas_entrada
        self.cantidad_neuronas_salida = cantidad_neuronas_salida
        self.tasa_aprendizaje = tasa_aprendizaje
#---------------------------------------------
# PREPARACIÓN DE LOS DATOS
#---------------------------------------------
    def datos(self):

        print("N.º columnas: ",len(self.observaciones.columns))
#Para el aprendizaje solo tomamos loa datos procedentes del sonar
        self.X = self.observaciones[self.observaciones.columns[0:60]].values

#Solo se toman los etiquetados
        y = self.observaciones[self.observaciones.columns[60]]

#Se codifica: Las minas son iguales a 0 y las rocas son iguales 1
        encoder = preprocessing.LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)

#Se añade un cifrado para crear clases:
# Si es una mina [1,0]
# Si es una roca [0,1]
        n_labels = len(y)
        n_unique_labels = len(np.unique(y))
        one_hot_encode = np.zeros((n_labels,n_unique_labels))
        one_hot_encode[np.arange(n_labels),y] = 1
        self.Y=one_hot_encode

        #Verificación tomando los registros 0 y 97
        print("Clase Roca:",int(self.Y[0][1]))
        print("Clase Mina:",int(self.Y[97][1]))


#---------------------------------------------
# CREACIÓN DE LOS CONJUNTOS DE APRENDIZAJE Y DE PRUEBAS
#---------------------------------------------
    def Mezcla(self):
#Mezclamos
        self.X, self.Y = shuffle(self.X, self.Y, random_state=1)

#Creación de los conjuntos de aprendizaje
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size=0.07, random_state=42)


#---------------------------------------------
# PARAMETRIZACIÓN DE LA RED NEURONAL
#---------------------------------------------
    def Parametrizacion(self):

        

        #Variable TensorFLow correspondiente a los 60 valores de las neuronas de entrada
        self.tf_neuronas_entradas_X = tf.placeholder(tf.float32,[None, 60])

        #Variable TensorFlow correspondiente a las 2 neuronas de salida
        self.tf_valores_reales_Y = tf.placeholder(tf.float32,[None, 2])


        self.pesos = {
        #60 neuronas de las entradas hacia 12 Neuronas de la capa oculta
        'capa_entrada_hacia_oculta': tf.Variable(tf.random_uniform([60, self.numeroNeuronas], minval=-0.3, maxval=0.3), tf.float32),

        # 12 neuronas de la capa oculta hacia 2 de la capa de salida
        'capa_oculta_hacia_salida': tf.Variable(tf.random_uniform([self.numeroNeuronas, 2], minval=-0.3, maxval=0.3), tf.float32),


        }

        self.peso_sesgo = {
            #1 sesgo de la capa de entrada hacia las 24 neuronas de la capa oculta
            'peso_sesgo_capa_entrada_hacia_oculta': tf.Variable(tf.zeros([self.numeroNeuronas]), tf.float32),

            #1 sesgo de la capa oculta hacia las 2 neuronas de la capa de salida
            'peso_sesgo_capa_oculta_hacia_salida': tf.Variable(tf.zeros([2]), tf.float32),
        }



#---------------------------------------------
# FUNCIÓN DE CREACIÓN DE LA RED NEURONAL
#---------------------------------------------




#---------------------------------------------
# CREACIÓN DE LA RED NEURONAL
#---------------------------------------------
    def RedNeuronal(self):
        self.red = red_neuronas_multicapa(self.tf_neuronas_entradas_X, self.pesos, self.peso_sesgo)


#---------------------------------------------
# ERROR Y OPTIMIZACIÓN
#---------------------------------------------

#Función de error de media cuadrática MSE
    def EyO(self):
        self.funcion_error = tf.reduce_sum(tf.pow(self.tf_valores_reales_Y-self.red,2))

#Función de precisión
        self.funcion_precision = tf.metrics.accuracy(labels=self.tf_valores_reales_Y,predictions=self.red)


#Descenso de gradiente con una tasa de aprendizaje fijada a 0,1
        self.optimizador = tf.train.GradientDescentOptimizer(learning_rate=self.tasa_aprendizaje).minimize(self.funcion_error)


#---------------------------------------------
# APRENDIZAJE
#---------------------------------------------
    def Aprendizaje(self):

        #Inicialización de la variable
        init = tf.global_variables_initializer()

        #Inicio de una sesión de aprendizaje
        self.sesion = tf.Session()
        self.sesion.run(init)

        #Para la realización de la gráfica para la MSE
        Grafica_MSE=[]


        #Para cada epoch
        for i in range(self.epochs):

            #Realización del aprendizaje con actualización de los pesos
            self.sesion.run(self.optimizador, feed_dict = {self.tf_neuronas_entradas_X: self.train_x, self.tf_valores_reales_Y:self.train_y})

            #Calcular el error de aprendizaje
            MSE = self.sesion.run(self.funcion_error, feed_dict = {self.tf_neuronas_entradas_X: self.train_x, self.tf_valores_reales_Y:self.train_y})

            #Visualización de la información
            Grafica_MSE.append(MSE)
            print("EPOCH (" + str(i) + "/" + str(self.epochs) + ") -  MSE: "+ str(MSE))


#Visualización gráfica MSE
        plt.plot(Grafica_MSE)
        plt.ylabel('MSE')
        plt.show()

    def Verificacion(self):
#---------------------------------------------
# VERIFICACIÓN DEL APRENDIZAJE
#---------------------------------------------

        #Las probabilidades de cada clase 'Mina' o 'roca' procedentes del aprendizaje se almacenan en el modelo.
        #Con la ayuda de tf.argmax, se recuperan los índices de las probabilidades más elevados para cada observación
        #Ejemplo: Si para una observación tenemos [0.56, 0.89] enviará 1 porque el valor más elevado se encuentra en el índice 1
        #Ejemplo: Si para una observación tenemos [0.90, 0.34] enviará 0 porque el valor más elevado se encuentra en el índice 0
        self.clasificaciones = tf.argmax(self.red, 1)

        #En la tabla de valores reales:
        #Las minas están codificadas como [1,0] y el índice que tiene el mayor valor es 0
        #Las rocas tienen el valor [0,1] y el índice que tiene el mayor valor es 1

        #Si la clasificación es [0.90, 0.34], el índice que tiene el mayor valor es 0
        #Si es una mina [1,0], el índice que tiene el mayor valor es 0
        #Si los dos índices son idénticos, entonces se puede afirmar que es una buena clasificación
        self.formula_calculo_clasificaciones_correctas = tf.equal(self.clasificaciones, tf.argmax(self.tf_valores_reales_Y,1))


        #La precisión se calcula haciendo la media (tf.mean)
        # de las clasificaciones buenas (después de haberlas convertido en decimales tf.cast, tf.float32)
        self.formula_precision = tf.reduce_mean(tf.cast(self.formula_calculo_clasificaciones_correctas, tf.float32))



#-------------------------------------------------------------------------
# PRECISIÓN EN LOS DATOS DE PRUEBAS
#-------------------------------------------------------------------------
    def PrecisionPruebas(self):
        n_clasificaciones = 0;
        n_clasificaciones_correctas = 0

#Se mira el conjunto de los datos de prueba (text_x)
        for i in range(0,self.test_x.shape[0]):

            #Se recupera la información
            datosSonar = self.test_x[i].reshape(1,60)
            clasificacionEsperada = self.test_y[i].reshape(1,2)

            # Se realiza la clasificación
            prediccion_run = self.sesion.run(self.clasificaciones, feed_dict={self.tf_neuronas_entradas_X:datosSonar})

            #Se calcula la precisión de la clasificación con la ayuda de la fórmula establecida antes
            accuracy_run = self.sesion.run(self.formula_precision, feed_dict={self.tf_neuronas_entradas_X:datosSonar, self.tf_valores_reales_Y:clasificacionEsperada})


            #Se muestra para observación la clase original y la clasificación realizada
            print(i,"Clase esperada: ", int( self.sesion.run(self.tf_valores_reales_Y[i][1],feed_dict={self.tf_valores_reales_Y:self.test_y})), "Clasificación: ", prediccion_run[0] )

            n_clasificaciones = n_clasificaciones+1
            if(accuracy_run*100 ==100):
                n_clasificaciones_correctas = n_clasificaciones_correctas+1


        print("-------------")
        print("Precisión en los datos de pruebas = "+str((n_clasificaciones_correctas/n_clasificaciones)*100)+"%")


#-------------------------------------------------------------------------
# PRECISIÓN EN LOS DATOS DE APRENDIZAJE
#-------------------------------------------------------------------------
    def PrecisionAprendizaje(self):
        n_clasificaciones = 0;
        n_clasificaciones_correctas = 0
        for i in range(0,self.train_x.shape[0]):
            # Recuperamos la información
            datosSonar = self.train_x[i].reshape(1, 60)
            clasificacionEsperada = self.train_y[i].reshape(1, 2)
            # Realizamos la clasificación
            prediccion_run = self.sesion.run(self.clasificaciones, feed_dict={self.tf_neuronas_entradas_X: datosSonar})

            # Calculamos la precisión de la clasificación con la ayuda de la fórmula establecida antes
            accuracy_run = self.sesion.run(self.formula_precision, feed_dict={self.tf_neuronas_entradas_X: datosSonar, self.tf_valores_reales_Y: clasificacionEsperada})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en los datos de aprendizaje = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")


#-------------------------------------------------------------------------
# PRECISIÓN EN EL CONJUNTO DE DATOS
#-------------------------------------------------------------------------

    def PrecisionDatos(self):
        n_clasificaciones = 0;
        n_clasificaciones_correctas = 0
        for i in range(0,207):

            prediccion_run = self.sesion.run(self.clasificaciones, feed_dict={self.tf_neuronas_entradas_X:self.X[i].reshape(1,60)})
            accuracy_run = self.sesion.run(self.formula_precision, feed_dict={self.tf_neuronas_entradas_X:self.X[i].reshape(1,60), self.tf_valores_reales_Y:self.Y[i].reshape(1,2)})

            n_clasificaciones = n_clasificaciones + 1
            if (accuracy_run * 100 == 100):
                n_clasificaciones_correctas = n_clasificaciones_correctas + 1


        print("Precisión en el conjunto de datos = " + str((n_clasificaciones_correctas / n_clasificaciones) * 100) + "%")




        self.sesion.close()

    @staticmethod
    def ejecutar():
        n=int(input("¿Cuantas neuronas quieres usar? (12,24,26,31)"))
        neu=Neuronas(N=n)
        neu.datos()
        neu.Mezcla()
        neu.Parametrizacion()
        neu.RedNeuronal()
        neu.EyO()
        neu.Aprendizaje()
        neu.Verificacion()
        neu.PrecisionPruebas()
        neu.PrecisionAprendizaje()
        neu.PrecisionDatos()

if __name__=="__main__":
    Neuronas.ejecutar()