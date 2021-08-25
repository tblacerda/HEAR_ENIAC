# SO
import os
import random
import shutil
import datetime
import sys
from shutil import copyfile

# Basicas
import numpy as np
import pandas as pd

# Gráficos
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('Agg')

# Audio
import librosa
import librosa.display

# TensorFlow
import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.backend import dropout
from keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras import Model

# ML Flow
from mlflow import log_metric, log_param, log_artifacts
import mlflow.tensorflow
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Metricas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Definicoes: Classe positiva (1) - Violencia
#             Classe negativa (0) - Não Violência

# Ignorar avisos de atualização, etc
import warnings
warnings.filterwarnings("ignore")
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%HH")
## Fim
user = 'root'
pwd = 'Celular135#'
hostname = '127.0.0.1'
port = '3306'
database = 'mlflow'
#uri = 'mysql://{'+ user + '}:{' + pwd + '}@{' + hostname + '}:3600/{' + database + '}'
#uri = 'mysql://root:Celular135#@localhost:3306/mlflow'
uri = 'mysql://root:Celular135#@localhost:3306/mlflow'
## TAGs do MLFlow
_MODEL_NAME_ = 'ResNet152V2 run2'
_MODEL_PATH_ = 'saved_model/' + timestamp +'/'
_MODEL_FULLPATH_  = _MODEL_PATH_ +_MODEL_NAME_

mlflow.set_tracking_uri('http://0.0.0.0:5000')
#mlflow.set_tracking_uri(uri)

#mlflow.set_tracking_uri('mysql://tiago:tiago@127.0.0.1:3306/mlflow')
mlflow.set_experiment(experiment_name='Artigo ENIAC')

tags = {
        "Projeto": "ENIAC Paper",
        "team": "Tiago B. Lacerda",
        "dataset": "HEAR DATASET REV2.4"
       }


DESIRED_ACCURACY = 0.995
# Localizacao das bases
EVALUATION_DIR = "/home/tiago/Documentos/__Programacao/repos/projeto-hear/HEAR_DATASET/Evaluation_SCAPER/"
#EVALUATION_DIR = "/home/tiago/Documentos/__Programacao/repos/projeto-hear/HEAR_DATASET/toy_eval/"
MODELOS_IMPORTADOS = "/home/tiago/Documentos/__Programacao/repos/projeto-hear/modelos_importados/"
DATASET_HEAR = "/home/tiago/Documentos/__Programacao/repos/projeto-hear/HEAR_DATASET/"

split_size = .9
TARGET_SIZE = (120,160)

def CheckTF():
    """ Checar se esta em um ambiente com GPU
    """
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    
    print(tf.version.VERSION)
    print('Diretorio de trabalho: ', os.getcwd())

def PrintComHora(texto):
    ''' print com a Hora de execucao
    '''
    hora = datetime.datetime.now().strftime('%H:%M:%S')
    print("[", hora, "]", texto)
    sys.stdout.flush()

def metricas(y_true, y_predict):
    acuracia = accuracy_score(y_true, y_predict)
    precision = precision_score(y_true, y_predict)
    recall = recall_score(y_true, y_predict)
    f1 = f1_score(y_true, y_predict)
    return acuracia, precision, recall, f1

def matriz_confusao(y_true, y_predict):
    matriz_conf = confusion_matrix(y_true, y_predict)
    fig = plt.figure()
    ax = plt.subplot()
    sns.heatmap(matriz_conf, annot=True, cmap='Blues', ax=ax)

    ax.set_xlabel('Valor Predito')
    ax.set_ylabel('Valor Real')
    ax.set_title('Matriz de Confusão') 
    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    plt.close()
    return fig

def ConferindoDir():

    print(len(os.listdir(DATASET_HEAR + 'Train/NAO_VIOLENCIA_PNG')))
    print(len(os.listdir(DATASET_HEAR + 'Train/VIOLENCIA_PNG')))
    print(len(os.listdir(DATASET_HEAR + 'Test/VIOLENCIA_PNG')))
    print(len(os.listdir(DATASET_HEAR + 'Test/NAO_VIOLENCIA_PNG')))
    print(len(os.listdir(DATASET_HEAR + 'Evaluation_SCAPER/VIOLENCIA_PNG')))
    print(len(os.listdir(DATASET_HEAR + 'Evaluation_SCAPER/NAO_VIOLENCIA_PNG')))


def ModeloKeras_ResNet(BATCH = 10, EPOCHs= 10):
    from tensorflow.keras.applications.resnet_v2 import ResNet152V2
    _MODEL_FILE_ = 'resnet152v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    
    PrintComHora('Inicio do treinamento do modelo...[  ]')
    # Create an instance of the inception model from the local pre-trained weights
    local_weights_file = MODELOS_IMPORTADOS + _MODEL_FILE_

    pre_trained_model = ResNet152V2(input_shape = (120, 160, 3), 
                                    include_top = False, 
                                    weights = None)

    pre_trained_model.load_weights(local_weights_file)

    # Make all the layers in the pre-trained model non-trainable
    for layer in pre_trained_model.layers:
        layer.trainable = False

    last_layer = pre_trained_model.get_layer('post_bn')
    last_output = last_layer.output

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    # Add a fully connected layer with 1,024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    # Add a dropout rate of 0.2
    x = layers.Dropout(0.2)(x)                  
    # Add a final sigmoid layer for classification
    x = layers.Dense  (2, activation='softmax')(x)

    model = Model(pre_trained_model.input, x)

    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model_summary = model.summary()

    train_generator, validation_generator = CallImageDataGenerator(batch=BATCH)
    meus_callbacks = Callbacks()

    history = model.fit(train_generator,
                              epochs= EPOCHs,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks=meus_callbacks)
    
    PrintComHora('Fim do treinamento do modelo...[OK!]')


    return model, history, model_summary


def CallImageDataGenerator(batch = 10):
    TRAINING_DIR = DATASET_HEAR + "Train/"
    train_datagen = ImageDataGenerator(rescale=1.0/255.)
    train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=batch,
                                                    class_mode='categorical',
                                                    target_size = TARGET_SIZE)
    print(train_generator.class_indices)
    VALIDATION_DIR = DATASET_HEAR + "Test/"
    validation_datagen = ImageDataGenerator(rescale=1.0/255.)
    validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=batch,
                                                              class_mode='categorical',
                                                              target_size=TARGET_SIZE)
    print(validation_generator.class_indices)                                                          
    return train_generator, validation_generator

def SalvarModelo(model):
    # Save the entire model as a SavedModel.
    
   # os.mkdir(diretorio)
    model.save(_MODEL_FULLPATH_)
    converter_tflite = tf.lite.TFLiteConverter.from_saved_model(_MODEL_FULLPATH_)
    tflite_model = converter_tflite.convert()
    with open(_MODEL_NAME_ + '.tflite', 'wb') as f:
        f.write(tflite_model)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>DESIRED_ACCURACY):
      print("\nReached 99.5% accuracy so cancelling training!")
      self.model.stop_training = True

def Callbacks():

    checkpoint_path = "training/" + timestamp + "/" + _MODEL_NAME_+ ".ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,                                                    
                                                    mode='max',
                                                    save_best_only=True)    

    val_loss_callback = ReduceLROnPlateau(monitor='val_accuracy',
                                          mode='max',
                                          min_delta=0.03,
                                          patience=3,
                                          factor=.5,
                                          min_lr=0.00001, 
                                          verbose=1)
    
    Callback_acc =  myCallback()


    return [cp_callback,Callback_acc]

    #return [cp_callback,val_loss_callback, Callback_acc]

def previsao(modelo, DirPositivo, DirNegativo):
    ''' Realiza a validacao do modelo
    '''
    ypred = []
    yreal = []
    FP, FN, TP, TN = 0,0,0,0
    
    for filename in os.listdir(DirPositivo):
        yreal.append(1)
        if "png" in filename:
            file_path = os.path.join(DirPositivo, filename)
            img = image.load_img(file_path, target_size=TARGET_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.
            images = np.vstack([x])
            classes = modelo.predict(images)
            if classes[0][1]>0.5:
                ypred.append(1)
                TP+=1
                print("\rTP: %i; FP: %i; TN: %i; FN: %i"%(TP,FP,TN,FN), end='')
            else:
                ypred.append(0)
                FP+=1   
                print("\rTP: %i; FP: %i; TN: %i; FN: %i"%(TP,FP,TN,FN), end='')

    for filename in os.listdir(DirNegativo):
        yreal.append(0)
        if "png" in filename:
            file_path = os.path.join(DirNegativo, filename)
            img = image.load_img(file_path, target_size=TARGET_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.
            images = np.vstack([x])
            classes = modelo.predict(images)
            if classes[0][1]<=0.5:
                ypred.append(0)
                TN+=1
                print("\rTP: %i; FP: %i; TN: %i; FN: %i"%(TP,FP,TN,FN), end='')
            else:
                ypred.append(1)
                FN+=1
                print("\rTP: %i; FP: %i; TN: %i; FN: %i"%(TP,FP,TN,FN), end='')

    return yreal, ypred

def main():
    CheckTF()    
    ConferindoDir()
    mlflow.tensorflow.autolog(every_n_iter=1)
    with mlflow.start_run(run_name=_MODEL_NAME_):
   
        #Registro das tags
        mlflow.set_tags(tags)

        #Criação do modelo
        model, history, summary = ModeloKeras_ResNet(BATCH = 10, EPOCHs = 40)
        model_uri = mlflow.get_artifact_uri("model")
        SalvarModelo(model)
        #PlotTreino(history)
        #Predição dos valores de testes
        y_true, y_pred =  previsao(model,
                        EVALUATION_DIR + "VIOLENCIA_PNG",
                        EVALUATION_DIR + "NAO_VIOLENCIA_PNG")

        #DataFrame
        df = pd.DataFrame({ 'Valor_Real':y_true, 
                            'Valor_Previsto':y_pred 
                            })
        temp_name = 'DataFrame.csv'
        df.to_csv(temp_name, index=False)
        mlflow.log_artifact(temp_name, "DataFrame")
        try:
            os.remove(temp_name)
        except FileNotFoundError as e:
            print(f"{temp_name} file is not found")
       
        #Métricas
        acuracia, precision, recall, f1 = metricas(y_true, y_pred)
        print("Acurácia: {}\nPrecision: {}\nRecall: {}\nF1-Score: {}".
                format(acuracia, precision, recall, f1))

        #Matriz de confusão
        matriz_conf = matriz_confusao(y_true, y_pred)
        temp_name = "confusion-matrix.png"
        matriz_conf.savefig(temp_name)
        mlflow.log_artifact(temp_name, "confusion-matrix-plots")
        try:
            os.remove(temp_name)
        except FileNotFoundError as e:
            print(f"{temp_name} file is not found")

        #Registro dos parâmetros e das métricas
        # mlflow.log_param("balanced", balanced)
        # mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("Acuracia", acuracia)
        mlflow.log_metric("Precision", precision)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("F1-Score", f1)

        #Registro do modelo
       
        
        # modelo ='./saved_model' + timestamp+ "/"

        # mlflow.tensorflow.log_model(modelo, "model_Inception")
        mlflow.log_artifact(local_path='./'+os.path.basename(__file__), artifact_path='code_models')
        
        #Registro do model_summary
        from contextlib import redirect_stdout
        temp_name = 'modelsummary.txt' 
        with open(temp_name, 'w') as f:
            with redirect_stdout(f):
                model.summary()
        mlflow.log_artifact(temp_name, 'Model Summary')
        try:
            os.remove(temp_name)    
        except FileNotFoundError as e:
            print(f"{temp_name} file is not found")

        mlflow.end_run()

if __name__ == "__main__":
    main()

