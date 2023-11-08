import numpy as np
import os
import seaborn as sn; sn.set(font_scale=1.4)
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


class_names=['combat','fire','destroyedbuilding','militaryvehicles','humanitarianaid']
c_n_label={class_names:i for i,class_names in enumerate(class_names)}

nb=len(class_names)
print(c_n_label)
I_s=(150,150)

def load_d():
  DI=r"D:\Data_dl"
  Ca=["Train","Test"]
  output=[]
  for cat in Ca:
    path=f'{DI}/{cat}'
    print(path)
    images=[]
    labels=[]
    print("Loading{}",format(cat))
    for fol in os.listdir(path):
      label=c_n_label[fol]
      for fil in os.listdir(os.path.join(path,fol)):
        img_p=os.path.join(os.path.join(path,fol),fil)
        image=cv2.imread(img_p)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image=cv2.resize(image,I_s)
        images.append(image)
        labels.append(label)
         
    images=np.array(images,dtype='float32')
    labels=np.array(labels,dtype='int32')
    output.append((images,labels))
  return output

(T_i,T_l),(t_i,t_l)=load_d()

T_i,T_l=shuffle(T_i,T_l,random_state=20)

def dis(class_names,images,labels):
    figsize=(20,20)
    fig=plt.figure(figsize=figsize)
    fig.suptitle("examples",fontsize=16)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].astype(np.uint8))
        plt.xlabel(class_names[labels[i]])
    plt.show()
dis(class_names,T_i,T_l)  

T_i=T_i/255.0

t_i=t_i/255.0

model=tf.keras.Sequential([tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
tf.keras.layers.MaxPooling2D(2,2),tf.keras.layers.Conv2D(32,(3,3),activation='relu'),tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),tf.keras.layers.Dense(128,activation=tf.nn.relu),tf.keras.layers.Dense(50,activation=tf.nn.softmax)])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

his=model.fit(T_i,T_l,batch_size=128,epochs=15,validation_split=0.2)

model.save("Trained_Weight.h5")
