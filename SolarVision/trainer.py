from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from SolarVision.data import get_data

class Trainer():
    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train

    def initialize_model(self):
        '''instanciate and return the CNN architecture with less than 150,000 params'''
        model = models.Sequential()

        model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape=(320,320,3)))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Dropout(rate=0.2))

        model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Dropout(rate=0.2))

        model.add(layers.Conv2D(128, kernel_size=(2,2), activation='relu', padding='same'))
        model.add(layers.Dropout(rate=0.3))
        model.add(layers.MaxPool2D(pool_size=(2,2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation='relu'))
        model.add(layers.Dropout(rate=0.4))
        model.add(layers.Dense(1, activation='sigmoid')) 

        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

        self.model = model 

    def model_fit(self):
        '''fit the model'''
        es = EarlyStopping(patience=20,restore_best_weights=True)

        history = self.model.fit(self.X_train, self.y_train,
                            validation_split=0.3,
                            epochs=200,
                            batch_size=32,
                            callbacks=[es],
                            verbose=0, )

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        evaluation = self.model.evaluate(self.X_test, self.y_test) 
        accuracy = evaluation[1]
        return accuracy 

if __name__ == "__main__":
    X_train, X_test, y_train, y_test =  get_data() 
    trainer = Trainer(X_train, X_test, y_train, y_test) 
    res = trainer.evaluate() 
    print(res)