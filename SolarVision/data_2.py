from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.cloud import storage

def get_data_2():

    storage_client = storage.Client()
    bucket_name = "solarvision-test"

    bucket = storage_client.bucket(bucket_name)

    train_dir = f'{bucket.name}/data/data/train_set/'
    val_dir = f'{bucket.name}/data/data/val_set/'
    test_dir = f'{bucket.name}/data/data/test_set/'
    
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
