import tensorflow as tf
from datos import datos

def eval_model(modelpath, filepath):

    train_dataset, validation_dataset, test_dataset = datos(filepath)
    model = tf.keras.models.load_model(modelpath)

    _, binary_acc = model.evaluate(test_dataset)
    print(f"Categorical accuracy on the test set: {round(binary_acc * 100, 2)}%.")


def main():
    model = eval_model('trained_model.h5', 'data_entrenamiento.csv')


if __name__ == "__main__":
    main()