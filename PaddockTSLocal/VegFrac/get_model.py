from tensorflow.lite.python.interpreter import Interpreter
from os.path import dirname
from os.path import join

def f(n: int):
    models_dir = join(dirname(__file__), 'resources')
    available_models = [
        join(models_dir, "fcModel_32x32x32.tflite"),
        join(models_dir, "fcModel_64x64x64.tflite"),
        join(models_dir, "fcModel_256x64x256.tflite"),
        join(models_dir, "fcModel_256x128x256.tflite")
    ]
    return Interpreter(model_path=available_models[n-1])

def t():
    model = f(4)

if __name__ == '__main__':
    t()