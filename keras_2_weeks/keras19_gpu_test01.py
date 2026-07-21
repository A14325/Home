import tensorflow as tf
print(tf.__version__)


gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)

if(gpus):
    print("GPU 돈다~")
    
else:
    print("GPU 없다~")


#cudnn 설치한 가상환경(tf291gpu)로 변환해야함

