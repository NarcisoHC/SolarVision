from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from SolarVision.data import get_data
import joblib
from SolarVision.mlflowbase import MLFlowBase
from SolarVision.params import MLFLOW_URI, EXPERIMENT_NAME #, STORAGE_LOCATION #gcp Jan/Wolfgang
from SolarVision.gcp import upload_model_to_gcp  # gcp Jan/Wolfgang

class Trainer(MLFlowBase):
    def __init__(self, X_train, X_test, y_train, y_test):
        '''initiate Class object'''
        super().__init__(
            EXPERIMENT_NAME,
            MLFLOW_URI
        )
        self.model = None
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train

    def initialize_model(self):
        '''instanciate, compile and return the CNN model'''
        model = models.Sequential()

        params_model = {'0_CNN': 32, '0_kernel': (3,3),\
                        '1_pooling': (2,2), 
                        '2_dropout': 0.2,
                        '3_CNN': 64, '3_kernel': (3,3),
                        '4_pooling': (2,2),
                        '5_dropout': 0.2,
                        '6_CNN': 128, '6_kernel': (2,2),
                        '7_dropout': 0.3,
                        '8_pooling': (2,2),
                        '9_flatten': 'flatten',
                        '10_dense': 100,
                        '11_dropout': 0.4,
                        '12_dense': 1}

        model.add(layers.Conv2D(params_model['0_CNN'], kernel_size=params_model['0_kernel'], activation='relu', padding='same', input_shape=(320,320,3)))
        model.add(layers.MaxPool2D(pool_size=params_model['1_pooling']))
        model.add(layers.Dropout(rate=params_model['2_dropout']))

        model.add(layers.Conv2D(params_model['3_CNN'], kernel_size=params_model['3_kernel'], activation='relu', padding='same'))
        model.add(layers.MaxPool2D(pool_size=params_model['4_pooling']))
        model.add(layers.Dropout(rate=params_model['5_dropout']))

        model.add(layers.Conv2D(params_model['6_CNN'], kernel_size=params_model['6_kernel'], activation='relu', padding='same'))
        model.add(layers.Dropout(rate=params_model['7_dropout']))
        model.add(layers.MaxPool2D(pool_size=params_model['8_pooling']))

        model.add(layers.Flatten())
        model.add(layers.Dense(params_model['10_dense'], activation='relu'))
        model.add(layers.Dropout(rate=params_model['11_dropout']))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        self.model = model

        self.mlflow_create_run()
        for k, v in params_model.items():
            self.mlflow_log_param(k, v)

    def model_fit(self):
        '''fit the model'''
        params_fit = {'patience': 20, 'validation_split': 0.3, 'epochs': 200, 'batch_size': 32}
        
        es = EarlyStopping(patience=params_fit['patience'] ,restore_best_weights=True)

        history = self.model.fit(X_train, y_train,
                        validation_split=params_fit['validation_split'],
                        epochs=params_fit['epochs'],
                        batch_size=params_fit['batch_size'],
                        callbacks=[es],
                        verbose=0)
        
        self.params_fit = params_fit
        
        self.mlflow_create_run()
        for k, v in params_fit.items():
            self.mlflow_log_param(k, v)

    def evaluate(self):
        '''evaluates the model on test data and return accuracy'''
        evaluation = self.model.evaluate(self.X_test, self.y_test) 
        self.accuracy = evaluation[1]
        
        self.mlflow_create_run()
        self.mlflow_log_metric("accuracy", self.accuracy)

    def save_model(self):
        '''Save the model into a .joblib format'''
        joblib.dump(self.model, 'model.joblib') # not required when we switch to gcp
        
        # save model to gcp
        upload_model_to_gcp()
        # print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")
        
    def train(self):
        '''get data, instanciate instance of Trainer class, initialize, compile and fit model,
        log parameters and metrics in mlflow, evaluate and save model'''
        # get data
        X_train, X_test, y_train, y_test =  get_data() 
        
        # instanciate class
        trainer = Trainer(X_train, X_test, y_train, y_test)
        
        # initiate and compile model
        trainer.initialize_model()
        
        # fit model
        trainer.model_fit()
        
        # evaluate model
        res = trainer.evaluate() 
        
        # save model
        trainer.save_model()
        
        # print model, to be deleted when everything works
        print(f'accuracy: {res}')
        
        return trainer

if __name__ == "__main__":
    X_train, X_test, y_train, y_test =  get_data() 
    trainer = Trainer(X_train, X_test, y_train, y_test)
    trainer.initialize_model()
    trainer.model_fit()
    res = trainer.evaluate()
    trainer.save_model()
    print(f'accuracy: {res}')
    