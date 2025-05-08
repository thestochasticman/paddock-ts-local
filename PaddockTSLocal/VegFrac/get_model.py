from tensorflow.lite.python.interpreter import Interpreter
from os.path import dirname
from os.path import join

def f():
    path = join(dirname(__file__), 'resources', 'fcModel_64x64x64.tflite')
    return Interpreter(model_path=path)

def t():
    model = f()
    print(type(model))

if __name__ == '__main__':
    t()