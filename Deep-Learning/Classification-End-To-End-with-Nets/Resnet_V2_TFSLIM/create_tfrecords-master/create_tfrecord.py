import random
import tensorflow as tf
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset

# =====DEFININDO OS ARGUMENTOS=====
flags = tf.app.flags

# Diretorio do Dataset
flags.DEFINE_string('dataset_dir', "/home/messias/TensorFlow/Resnet_V2_TFSLIM/datasets", 'String: Your dataset directory')

# Proporcao do conjunto que sera utilizada para validacao (0.2 = 20%)
flags.DEFINE_float('validation_size', 0.2, 'Float: The proportion of examples in the dataset to be used for validation')

# Numero de fragmentos que sera quebrado os arquivos TFRecords
flags.DEFINE_integer('num_shards', 1, 'Int: Number of shards to split the TFRecord files')

# Semente para repetitividade
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

# Nome que os arquivos de saida .tfrecord receberao
flags.DEFINE_string('tfrecord_filename', "Mammoset", 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

def main():

    #==============================================================CHECKS==========================================================================
    #Check if there is a tfrecord_filename entered
    if not FLAGS.tfrecord_filename:
        raise ValueError('Nao foi definido o nome de arquivo.')

    #Check if there is a dataset directory entered
    if not FLAGS.dataset_dir:
        raise ValueError('Nao foi definido o diretorio do dataset.')

    #If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_dir = FLAGS.dataset_dir, _NUM_SHARDS = FLAGS.num_shards, output_filename = FLAGS.tfrecord_filename):
        print 'Arquivos do Datasets ja existem. Finalizando sem recria-los.'
        return None
    #==============================================================END OF CHECKS===================================================================

    #Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)

    #Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    #Find the number of validation examples we need
    num_validation = int(FLAGS.validation_size * len(photo_filenames))

    # Divide the training datasets into train and test:
    random.seed(FLAGS.random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, FLAGS.dataset_dir)

    print '\nTerminado de converter o %s dataset!' % (FLAGS.tfrecord_filename)

if __name__ == "__main__":
    main()
