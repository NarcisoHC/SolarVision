from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_2():

    train_dir = 'gs://solarvision-test/data/data/train_set/'
    val_dir = 'gs://solarvision-test/data/data/val_set/'
    test_dir = 'gs://solarvision-test/data/data/test_set/'
    
    datagen = ImageDataGenerator(rescale=1./255)
    target_size = (320, 320)

    train_dataset = datagen.flow_from_directory(train_dir, 
                                                class_mode='binary',
                                                batch_size=32,
                                                target_size=target_size)

    val_dataset = datagen.flow_from_directory(val_dir, 
                                              class_mode='binary',
                                              batch_size=32,
                                              target_size=target_size)

    test_dataset = datagen.flow_from_directory(test_dir, 
                                               class_mode='binary',
                                               batch_size=32,
                                               target_size=target_size)
    
    return train_dataset, val_dataset, test_dataset
