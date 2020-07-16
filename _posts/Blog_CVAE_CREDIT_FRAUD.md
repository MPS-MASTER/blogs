# Solving Data Imbalance  Problem using Generative modelling(CONDITIONAL VARIATIONAL AUTOENCODERS)



# Introduction : 
Do you want to achieve more than 90% f1-score, Do you want to have a strong and reliable Deep Learning solution for your fraud detection or anomaly detection. You tried all the techniques but didn't get the desired results. Here I am gonna present you the generative way of solving data imbalance problem. We will generate new data points for minority class and use those data points to fill some of the gaps of imbalance class.
## Problem I am gonna solve :
Most of the time you don't get the good results for highly skewed or highly imbalance dataset. In these cases we use precision and recall as accuracy measure.  Here I have used "creditfraud" data from kaggle  to show the generative way of handling class imbalance. Also compared the results obtained by popular techniques vs generative modeling for handling class imbalance.
## Dataset :
You can download from below kaggle link:


https://www.kaggle.com/mlg-ulb/creditcardfraud
## Previous Best Results obtained by different Techniques :

### Accuracieds taken from different solutions on kaggle

Accuracy on Under sampled data :

Logistic Regression:  0.9798658657817729 

KNears Neighbors:  0.9246195096680248 

Support Vector Classifier:  0.9746783159014857 

Decision Tree Classifier:  0.9173877431007686
### Since Logistic Regression has better accuracy we choose it for further prediction


Logistic regression Precision and recall:



Recall Score: 0.90 

Precision Score: 0.76

 F1 Score: 0.82 

Accuracy Score: 0.81


### Some of them got 93% Recall score but very less precision score

### So best is 93% recall and around 98% validation score

## Now I will Use Generative method to improve accuracies :
--------------------------- Start with coding part
    
    ------------------------------ IMPORTS
```python
import warnings
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate as concat
from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
#from scipy.misc import imsave
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
%pylab inline
```

    Using TensorFlow backend.
    

    Populating the interactive namespace from numpy and matplotlib
    

```python
import pandas as pd
```

```python
import random
random.seed(150)  # for generating same random no always
```

```python


data = pd.read_csv(r"C:\Users\mpspa\Desktop\Kaggle\creditcard.csv\creditcard.csv")

print("Shape :", data.shape)
print("Columns :",data.columns)
columns = data.columns
data.head(3)
```

    Shape : (284807, 31)
    Columns : Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
           'Class'],
          dtype='object')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 31 columns</p>
</div>



#### 30 features and 1 output column(Class)
Lets see data distribution with different class
```python
data.Class.value_counts()
```




    0    284315
    1       492
    Name: Class, dtype: int64



```python
print(" class 0 percentage : ", 100*284315/284807)
print(" class 1 percentage : ", 100*492/284807)
```

     class 0 percentage :  99.827251436938
     class 1 percentage :  0.1727485630620034
    

#### Class 1 is fraud class and its only .17% , Data is higly imbalance or skewed

```python
import seaborn as sns
sns.countplot(data.Class)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1e258df5708>




![png](/images/Blog_CVAE_CREDIT_FRAUD_files/output_24_1.png)


### Prerequisite to know before looking at this code

- Pandas 
- Numpy 
- Neural Networks
- Keras 
- Autoencoders

#### For variational auto encoder you can refer to this video : https://www.youtube.com/watch?v=W4peyiOaEFU

- We will generate data for class 1
- Use that data to make better fraud detection classifier


## I will divide my code in 6 parts

1. Data Preparation
1. Encoder
2. Latent Space
3. Decoder
4. Reconstrution of new fraud data
5. Accuracy calculation and Evaluation of fraud detector model

### Data Preparation
#### Since i am generating data for class 1 , i will train my model on class 1 only

```python
data1 = data[data.Class ==1]
```

```python
data1.Class.value_counts()
```




    1    492
    Name: Class, dtype: int64


------------------Normalizing data --------------------


```python
from sklearn.preprocessing import Normalizer
```

```python
ssc = Normalizer()
```

```python
data1.iloc[:, :30] = ssc.fit_transform(data1.iloc[:,:30])
```

```python
data1 = pd.DataFrame(data1)
```

```python
fraud_data = data1
fraud_data = fraud_data.iloc[:450, :]

fraud_x = fraud_data.drop("Class", axis =1)
fraud_y = fraud_data.Class
```

```python
fraud_y.shape
```




    (450,)


--------------- Breaking in train and Test -------------------------
--------------- 400 train and 50 test data ----------------------------


```python
train_x = fraud_x.iloc[:400, :]
train_y =  fraud_y.iloc[:400]
test_x  = fraud_x.iloc[400:450, :]
test_y =  fraud_y.iloc[400: 450]
```

```python
train_y.shape, test_y.shape, train_x.shape, test_x.shape
```




    ((400,), (50,), (400, 30), (50, 30))



## Encoder

- We will make encoder
- We sample encoder output as normal distribution and it will be the latent space 
- We'll be using the Keras functional API rather than the sequential because of the slightly more complex structure of the VAE. 
- First we'll explicitly define __input__ layers for __X__ and __y__. 

- Keras needs to know their shapes at the input layer, but can infer them later on.


```python
m   = 50 # batch size
n_z = 2   # latent space size

encoder_dim1 = 16 # dim of encoder hidden layer
decoder_dim  = 16 # dim of decoder hidden layer

decoder_out_dim = 30 # dim of decoder output layer

activ = 'relu'
optim = Adam(lr=0.001)

n_x = 30  # Input feature dimention
n_y = 1   # Output dimention( its 1 because of binary output 0 and 1)

n_epoch = 100
```

```python
X     = Input(shape=(n_x,))
label = Input(shape=(n_y,))
```

- Next we'll concatenate the X and y vectors. 

- It may appear that it would've been simpler to merge the pixel and class label vectors from the beginning (now that they're both 1d) rather than reading them into the graph as separate input layers and concatenating them... but in reality, we need them to remain separate entities so that we can properly calculate our reconstruction error (we aren't asking the autoencoder to reassemble y in addition to X).

```python
inputs = concat([X, label])
```

```python
encoder_h = Dense(encoder_dim1, activation=activ)(inputs)

mu      = Dense(n_z, activation='linear')(encoder_h)
l_sigma = Dense(n_z, activation='linear')(encoder_h)
```

## Latent Space

- Now that we've built our encoder and defined our sampling function, our latent space (z) is easy to define.

- First, using our sample_z function, we generate a vector of length n_z (in this case 2). If this were a normal VAE we could stop here and move on to the decoder, but instead we are going to concatenate our latent z respresentation with the same sparse y vector that we initially merged to our pixel representation X in the input layers. This gives us a 1x12 vector with 3 non-zero values as we move from the latent space to the decoder.


```python
def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(l_sigma / 2) * eps

# Sampling latent space
#z = Lambda(sample_z, output_shape = (n_z, ))([mu, l_sigma])
```

```python
# Sampling latent space
z = Lambda(sample_z, output_shape = (n_z, ))([mu, l_sigma])


```

- Now we add labels to latest space
- It helps you to give your desirable input and get generated data for that
- z is style learned  and label part is data we will give 




```python
# merge latent space with label
zc = concat([z, label])
```

- there will be n_z+1 values that will be non zero
- and will be passed to decoder  




## Decoder 
- The encoder has hopefully taken the information contained in 30 features (plus the class label), and created some vector z. The decoding process is the reconstruction from z to X_hat. Unlike a normal undercomplete autoencoder, we won't stick to a rigid symmetrical funnel-type architecture here. Instead I'll define two dense layers of 16 and 30 neurons that have ReLU and sigmoidal activation functions, respectively.

```python
decoder_hidden = Dense(decoder_dim, activation=activ)
decoder_out    = Dense(decoder_out_dim, activation='sigmoid')

h_p     = decoder_hidden(zc)
outputs = decoder_out(h_p)
```

### Defining the loss


#### Two types of loss
1. Kl divergence
2. Reconstruction



- If you're familiar with autoencoders, you probably understand that they are backpropagated using reconstruction loss. This is a measure of error between the input X and the decoded output X_hat. In VAEs, our loss is the sum of reconstruction error and the kullback-leibler divergence between our $\mu$ and log-$\sigma$ and the standard normal.

- In this notebook I've defined the vae_loss function, which we'll use to optimize our model. I've also broken it down into the KL_loss and recon_loss subcomponents so that we can track these values as metrics during training.

```python
def vae_loss(y_true, y_pred):

    #recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)

    return recon + kl

def KL_loss(y_true, y_pred):
    return(0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

def recon_loss(y_true, y_pred):
    print(y_true)
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
```

#### Create CVAE, Encoder, Decoder layers



```python
cvae    = Model([X, label], outputs)
encoder = Model([X, label], mu)

d_in = Input(shape=(n_z+n_y,))
d_h  = decoder_hidden(d_in)
d_out   = decoder_out(d_h)
decoder = Model(d_in, d_out)
```
-------------------- Compile the model


```python
cvae.compile(optimizer=optim, 
             loss=vae_loss, 
             metrics = [KL_loss, recon_loss])
```

    Tensor("dense_5_target:0", shape=(None, None), dtype=float32)
    
-------------------- Fit and train the model


```python
# compile and fit
cvae_hist = cvae.fit([np.array(train_x), np.array(train_y).reshape(len(train_y),1)], 
                     np.array(train_x), 
                     verbose = 1, 
                     batch_size=m, 
                     epochs=n_epoch,
                     validation_data= ([np.array(test_x), np.array(test_y).reshape(len(test_y),1)], 
                     np.array(test_x)),
                     callbacks = [EarlyStopping(patience = 10)])
```

    Train on 400 samples, validate on 50 samples
    Epoch 1/100
    400/400 [==============================] - 2s 6ms/step - loss: 20.2842 - KL_loss: 0.0746 - recon_loss: 20.2097 - val_loss: 19.9961 - val_KL_loss: 0.0643 - val_recon_loss: 19.9318
    Epoch 2/100
    400/400 [==============================] - 0s 220us/step - loss: 19.7711 - KL_loss: 0.0569 - recon_loss: 19.7141 - val_loss: 19.4787 - val_KL_loss: 0.0483 - val_recon_loss: 19.4304
    Epoch 3/100
    400/400 [==============================] - 0s 200us/step - loss: 19.2467 - KL_loss: 0.0427 - recon_loss: 19.2039 - val_loss: 18.9426 - val_KL_loss: 0.0366 - val_recon_loss: 18.9060
    Epoch 4/100
    400/400 [==============================] - 0s 213us/step - loss: 18.6999 - KL_loss: 0.0332 - recon_loss: 18.6667 - val_loss: 18.3796 - val_KL_loss: 0.0303 - val_recon_loss: 18.3493
    Epoch 5/100
    400/400 [==============================] - 0s 208us/step - loss: 18.1218 - KL_loss: 0.0300 - recon_loss: 18.0918 - val_loss: 17.7796 - val_KL_loss: 0.0314 - val_recon_loss: 17.7483
    Epoch 6/100
    400/400 [==============================] - 0s 203us/step - loss: 17.5044 - KL_loss: 0.0343 - recon_loss: 17.4701 - val_loss: 17.1419 - val_KL_loss: 0.0395 - val_recon_loss: 17.1024
    Epoch 7/100
    400/400 [==============================] - 0s 215us/step - loss: 16.8493 - KL_loss: 0.0455 - recon_loss: 16.8037 - val_loss: 16.4617 - val_KL_loss: 0.0551 - val_recon_loss: 16.4066
    Epoch 8/100
    400/400 [==============================] - 0s 260us/step - loss: 16.1495 - KL_loss: 0.0649 - recon_loss: 16.0846 - val_loss: 15.7312 - val_KL_loss: 0.0796 - val_recon_loss: 15.6516
    Epoch 9/100
    400/400 [==============================] - 0s 235us/step - loss: 15.3989 - KL_loss: 0.0938 - recon_loss: 15.3051 - val_loss: 14.9553 - val_KL_loss: 0.1143 - val_recon_loss: 14.8411
    Epoch 10/100
    400/400 [==============================] - 0s 320us/step - loss: 14.6026 - KL_loss: 0.1334 - recon_loss: 14.4692 - val_loss: 14.1321 - val_KL_loss: 0.1609 - val_recon_loss: 13.9713
    Epoch 11/100
    400/400 [==============================] - 0s 348us/step - loss: 13.7606 - KL_loss: 0.1860 - recon_loss: 13.5745 - val_loss: 13.2718 - val_KL_loss: 0.2214 - val_recon_loss: 13.0504
    Epoch 12/100
    400/400 [==============================] - 0s 486us/step - loss: 12.8885 - KL_loss: 0.2525 - recon_loss: 12.6360 - val_loss: 12.3872 - val_KL_loss: 0.2953 - val_recon_loss: 12.0919
    Epoch 13/100
    400/400 [==============================] - 0s 240us/step - loss: 11.9975 - KL_loss: 0.3321 - recon_loss: 11.6653 - val_loss: 11.4889 - val_KL_loss: 0.3823 - val_recon_loss: 11.1066
    Epoch 14/100
    400/400 [==============================] - 0s 165us/step - loss: 11.0983 - KL_loss: 0.4242 - recon_loss: 10.6741 - val_loss: 10.5951 - val_KL_loss: 0.4802 - val_recon_loss: 10.1150
    Epoch 15/100
    400/400 [==============================] - 0s 203us/step - loss: 10.2165 - KL_loss: 0.5243 - recon_loss: 9.6922 - val_loss: 9.7285 - val_KL_loss: 0.5812 - val_recon_loss: 9.1473
    Epoch 16/100
    400/400 [==============================] - 0s 203us/step - loss: 9.3672 - KL_loss: 0.6233 - recon_loss: 8.7439 - val_loss: 8.9060 - val_KL_loss: 0.6757 - val_recon_loss: 8.2304
    Epoch 17/100
    400/400 [==============================] - 0s 233us/step - loss: 8.5741 - KL_loss: 0.7106 - recon_loss: 7.8636 - val_loss: 8.1375 - val_KL_loss: 0.7510 - val_recon_loss: 7.3865
    Epoch 18/100
    400/400 [==============================] - 0s 228us/step - loss: 7.8294 - KL_loss: 0.7736 - recon_loss: 7.0558 - val_loss: 7.4268 - val_KL_loss: 0.7974 - val_recon_loss: 6.6294
    Epoch 19/100
    400/400 [==============================] - 0s 220us/step - loss: 7.1426 - KL_loss: 0.8074 - recon_loss: 6.3352 - val_loss: 6.7715 - val_KL_loss: 0.8165 - val_recon_loss: 5.9550
    Epoch 20/100
    400/400 [==============================] - 0s 200us/step - loss: 6.5109 - KL_loss: 0.8174 - recon_loss: 5.6935 - val_loss: 6.1796 - val_KL_loss: 0.8163 - val_recon_loss: 5.3633
    Epoch 21/100
    400/400 [==============================] - 0s 193us/step - loss: 5.9512 - KL_loss: 0.8124 - recon_loss: 5.1388 - val_loss: 5.6422 - val_KL_loss: 0.8067 - val_recon_loss: 4.8355
    Epoch 22/100
    400/400 [==============================] - 0s 195us/step - loss: 5.4245 - KL_loss: 0.8008 - recon_loss: 4.6237 - val_loss: 5.1421 - val_KL_loss: 0.7937 - val_recon_loss: 4.3483
    Epoch 23/100
    400/400 [==============================] - 0s 245us/step - loss: 4.9434 - KL_loss: 0.7880 - recon_loss: 4.1553 - val_loss: 4.6752 - val_KL_loss: 0.7819 - val_recon_loss: 3.8933
    Epoch 24/100
    400/400 [==============================] - 0s 314us/step - loss: 4.4975 - KL_loss: 0.7776 - recon_loss: 3.7199 - val_loss: 4.2537 - val_KL_loss: 0.7729 - val_recon_loss: 3.4808
    Epoch 25/100
    400/400 [==============================] - 0s 378us/step - loss: 4.0962 - KL_loss: 0.7690 - recon_loss: 3.3272 - val_loss: 3.8757 - val_KL_loss: 0.7640 - val_recon_loss: 3.1117
    Epoch 26/100
    400/400 [==============================] - 0s 433us/step - loss: 3.7319 - KL_loss: 0.7600 - recon_loss: 2.9719 - val_loss: 3.5329 - val_KL_loss: 0.7557 - val_recon_loss: 2.7772
    Epoch 27/100
    400/400 [==============================] - 0s 255us/step - loss: 3.4095 - KL_loss: 0.7531 - recon_loss: 2.6564 - val_loss: 3.2242 - val_KL_loss: 0.7504 - val_recon_loss: 2.4738
    Epoch 28/100
    400/400 [==============================] - 0s 168us/step - loss: 3.1125 - KL_loss: 0.7487 - recon_loss: 2.3638 - val_loss: 2.9485 - val_KL_loss: 0.7468 - val_recon_loss: 2.2016
    Epoch 29/100
    400/400 [==============================] - 0s 168us/step - loss: 2.8501 - KL_loss: 0.7455 - recon_loss: 2.1046 - val_loss: 2.7054 - val_KL_loss: 0.7442 - val_recon_loss: 1.9612
    Epoch 30/100
    400/400 [==============================] - 0s 188us/step - loss: 2.6211 - KL_loss: 0.7432 - recon_loss: 1.8778 - val_loss: 2.4993 - val_KL_loss: 0.7416 - val_recon_loss: 1.7577
    Epoch 31/100
    400/400 [==============================] - 0s 278us/step - loss: 2.4319 - KL_loss: 0.7392 - recon_loss: 1.6927 - val_loss: 2.3279 - val_KL_loss: 0.7351 - val_recon_loss: 1.5927
    Epoch 32/100
    400/400 [==============================] - 0s 283us/step - loss: 2.2757 - KL_loss: 0.7304 - recon_loss: 1.5453 - val_loss: 2.1832 - val_KL_loss: 0.7229 - val_recon_loss: 1.4602
    Epoch 33/100
    400/400 [==============================] - 0s 260us/step - loss: 2.1379 - KL_loss: 0.7154 - recon_loss: 1.4225 - val_loss: 2.0590 - val_KL_loss: 0.7045 - val_recon_loss: 1.3545
    Epoch 34/100
    400/400 [==============================] - 0s 263us/step - loss: 2.0285 - KL_loss: 0.6957 - recon_loss: 1.3328 - val_loss: 1.9488 - val_KL_loss: 0.6844 - val_recon_loss: 1.2644
    Epoch 35/100
    400/400 [==============================] - 0s 214us/step - loss: 1.9203 - KL_loss: 0.6764 - recon_loss: 1.2439 - val_loss: 1.8502 - val_KL_loss: 0.6660 - val_recon_loss: 1.1841
    Epoch 36/100
    400/400 [==============================] - 0s 215us/step - loss: 1.8255 - KL_loss: 0.6582 - recon_loss: 1.1672 - val_loss: 1.7618 - val_KL_loss: 0.6479 - val_recon_loss: 1.1139
    Epoch 37/100
    400/400 [==============================] - 0s 208us/step - loss: 1.7422 - KL_loss: 0.6397 - recon_loss: 1.1025 - val_loss: 1.6819 - val_KL_loss: 0.6290 - val_recon_loss: 1.0529
    Epoch 38/100
    400/400 [==============================] - 0s 240us/step - loss: 1.6668 - KL_loss: 0.6206 - recon_loss: 1.0462 - val_loss: 1.6091 - val_KL_loss: 0.6100 - val_recon_loss: 0.9991
    Epoch 39/100
    400/400 [==============================] - 0s 280us/step - loss: 1.5977 - KL_loss: 0.6016 - recon_loss: 0.9960 - val_loss: 1.5423 - val_KL_loss: 0.5910 - val_recon_loss: 0.9513
    Epoch 40/100
    400/400 [==============================] - 0s 393us/step - loss: 1.5323 - KL_loss: 0.5823 - recon_loss: 0.9500 - val_loss: 1.4801 - val_KL_loss: 0.5714 - val_recon_loss: 0.9087
    Epoch 41/100
    400/400 [==============================] - ETA: 0s - loss: 1.4684 - KL_loss: 0.5664 - recon_loss: 0.90 - 0s 348us/step - loss: 1.4709 - KL_loss: 0.5623 - recon_loss: 0.9086 - val_loss: 1.4222 - val_KL_loss: 0.5512 - val_recon_loss: 0.8710
    Epoch 42/100
    400/400 [==============================] - 0s 233us/step - loss: 1.4158 - KL_loss: 0.5422 - recon_loss: 0.8736 - val_loss: 1.3683 - val_KL_loss: 0.5313 - val_recon_loss: 0.8371
    Epoch 43/100
    400/400 [==============================] - 0s 168us/step - loss: 1.3647 - KL_loss: 0.5226 - recon_loss: 0.8421 - val_loss: 1.3181 - val_KL_loss: 0.5123 - val_recon_loss: 0.8057
    Epoch 44/100
    400/400 [==============================] - 0s 163us/step - loss: 1.3099 - KL_loss: 0.5041 - recon_loss: 0.8058 - val_loss: 1.2708 - val_KL_loss: 0.4943 - val_recon_loss: 0.7765
    Epoch 45/100
    400/400 [==============================] - 0s 283us/step - loss: 1.2688 - KL_loss: 0.4866 - recon_loss: 0.7823 - val_loss: 1.2262 - val_KL_loss: 0.4773 - val_recon_loss: 0.7489
    Epoch 46/100
    400/400 [==============================] - 0s 273us/step - loss: 1.2230 - KL_loss: 0.4699 - recon_loss: 0.7531 - val_loss: 1.1841 - val_KL_loss: 0.4610 - val_recon_loss: 0.7231
    Epoch 47/100
    400/400 [==============================] - 0s 270us/step - loss: 1.1812 - KL_loss: 0.4538 - recon_loss: 0.7275 - val_loss: 1.1440 - val_KL_loss: 0.4451 - val_recon_loss: 0.6989
    Epoch 48/100
    400/400 [==============================] - 0s 253us/step - loss: 1.1370 - KL_loss: 0.4380 - recon_loss: 0.6991 - val_loss: 1.1058 - val_KL_loss: 0.4294 - val_recon_loss: 0.6764
    Epoch 49/100
    400/400 [==============================] - 0s 214us/step - loss: 1.1101 - KL_loss: 0.4224 - recon_loss: 0.6877 - val_loss: 1.0693 - val_KL_loss: 0.4139 - val_recon_loss: 0.6554
    Epoch 50/100
    400/400 [==============================] - 0s 262us/step - loss: 1.0724 - KL_loss: 0.4069 - recon_loss: 0.6655 - val_loss: 1.0345 - val_KL_loss: 0.3986 - val_recon_loss: 0.6359
    Epoch 51/100
    400/400 [==============================] - 0s 298us/step - loss: 1.0399 - KL_loss: 0.3920 - recon_loss: 0.6479 - val_loss: 1.0012 - val_KL_loss: 0.3841 - val_recon_loss: 0.6171
    Epoch 52/100
    400/400 [==============================] - 0s 245us/step - loss: 1.0024 - KL_loss: 0.3777 - recon_loss: 0.6247 - val_loss: 0.9691 - val_KL_loss: 0.3701 - val_recon_loss: 0.5990
    Epoch 53/100
    400/400 [==============================] - 0s 208us/step - loss: 0.9755 - KL_loss: 0.3638 - recon_loss: 0.6117 - val_loss: 0.9383 - val_KL_loss: 0.3564 - val_recon_loss: 0.5818
    Epoch 54/100
    400/400 [==============================] - 0s 208us/step - loss: 0.9386 - KL_loss: 0.3504 - recon_loss: 0.5881 - val_loss: 0.9085 - val_KL_loss: 0.3433 - val_recon_loss: 0.5652
    Epoch 55/100
    400/400 [==============================] - 0s 190us/step - loss: 0.9081 - KL_loss: 0.3375 - recon_loss: 0.5706 - val_loss: 0.8795 - val_KL_loss: 0.3304 - val_recon_loss: 0.5491
    Epoch 56/100
    400/400 [==============================] - 0s 190us/step - loss: 0.8868 - KL_loss: 0.3244 - recon_loss: 0.5624 - val_loss: 0.8513 - val_KL_loss: 0.3171 - val_recon_loss: 0.5342
    Epoch 57/100
    400/400 [==============================] - 0s 170us/step - loss: 0.8592 - KL_loss: 0.3111 - recon_loss: 0.5482 - val_loss: 0.8242 - val_KL_loss: 0.3039 - val_recon_loss: 0.5202
    Epoch 58/100
    400/400 [==============================] - 0s 175us/step - loss: 0.8268 - KL_loss: 0.2982 - recon_loss: 0.5286 - val_loss: 0.7979 - val_KL_loss: 0.2914 - val_recon_loss: 0.5065
    Epoch 59/100
    400/400 [==============================] - 0s 288us/step - loss: 0.8040 - KL_loss: 0.2860 - recon_loss: 0.5180 - val_loss: 0.7725 - val_KL_loss: 0.2796 - val_recon_loss: 0.4928
    Epoch 60/100
    400/400 [==============================] - 0s 263us/step - loss: 0.7776 - KL_loss: 0.2746 - recon_loss: 0.5031 - val_loss: 0.7479 - val_KL_loss: 0.2686 - val_recon_loss: 0.4793
    Epoch 61/100
    400/400 [==============================] - 0s 261us/step - loss: 0.7527 - KL_loss: 0.2637 - recon_loss: 0.4890 - val_loss: 0.7241 - val_KL_loss: 0.2580 - val_recon_loss: 0.4662
    Epoch 62/100
    400/400 [==============================] - 0s 250us/step - loss: 0.7326 - KL_loss: 0.2533 - recon_loss: 0.4793 - val_loss: 0.7011 - val_KL_loss: 0.2479 - val_recon_loss: 0.4532
    Epoch 63/100
    400/400 [==============================] - 0s 165us/step - loss: 0.7065 - KL_loss: 0.2433 - recon_loss: 0.4632 - val_loss: 0.6788 - val_KL_loss: 0.2380 - val_recon_loss: 0.4408
    Epoch 64/100
    400/400 [==============================] - 0s 187us/step - loss: 0.6870 - KL_loss: 0.2336 - recon_loss: 0.4534 - val_loss: 0.6572 - val_KL_loss: 0.2285 - val_recon_loss: 0.4287
    Epoch 65/100
    400/400 [==============================] - 0s 203us/step - loss: 0.6672 - KL_loss: 0.2243 - recon_loss: 0.4429 - val_loss: 0.6362 - val_KL_loss: 0.2193 - val_recon_loss: 0.4170
    Epoch 66/100
    400/400 [==============================] - 0s 270us/step - loss: 0.6479 - KL_loss: 0.2151 - recon_loss: 0.4329 - val_loss: 0.6159 - val_KL_loss: 0.2102 - val_recon_loss: 0.4057
    Epoch 67/100
    400/400 [==============================] - 0s 263us/step - loss: 0.6262 - KL_loss: 0.2060 - recon_loss: 0.4202 - val_loss: 0.5962 - val_KL_loss: 0.2013 - val_recon_loss: 0.3948
    Epoch 68/100
    400/400 [==============================] - 0s 230us/step - loss: 0.6073 - KL_loss: 0.1974 - recon_loss: 0.4099 - val_loss: 0.5770 - val_KL_loss: 0.1928 - val_recon_loss: 0.3841
    Epoch 69/100
    400/400 [==============================] - 0s 208us/step - loss: 0.5882 - KL_loss: 0.1890 - recon_loss: 0.3992 - val_loss: 0.5584 - val_KL_loss: 0.1846 - val_recon_loss: 0.3737
    Epoch 70/100
    400/400 [==============================] - 0s 200us/step - loss: 0.5701 - KL_loss: 0.1809 - recon_loss: 0.3893 - val_loss: 0.5403 - val_KL_loss: 0.1766 - val_recon_loss: 0.3637
    Epoch 71/100
    400/400 [==============================] - 0s 188us/step - loss: 0.5534 - KL_loss: 0.1729 - recon_loss: 0.3805 - val_loss: 0.5227 - val_KL_loss: 0.1687 - val_recon_loss: 0.3540
    Epoch 72/100
    400/400 [==============================] - 0s 193us/step - loss: 0.5310 - KL_loss: 0.1652 - recon_loss: 0.3657 - val_loss: 0.5057 - val_KL_loss: 0.1612 - val_recon_loss: 0.3445
    Epoch 73/100
    400/400 [==============================] - 0s 200us/step - loss: 0.5161 - KL_loss: 0.1578 - recon_loss: 0.3583 - val_loss: 0.4892 - val_KL_loss: 0.1539 - val_recon_loss: 0.3352
    Epoch 74/100
    400/400 [==============================] - 0s 245us/step - loss: 0.5020 - KL_loss: 0.1507 - recon_loss: 0.3513 - val_loss: 0.4731 - val_KL_loss: 0.1470 - val_recon_loss: 0.3261
    Epoch 75/100
    400/400 [==============================] - 0s 311us/step - loss: 0.4820 - KL_loss: 0.1438 - recon_loss: 0.3382 - val_loss: 0.4575 - val_KL_loss: 0.1402 - val_recon_loss: 0.3173
    Epoch 76/100
    400/400 [==============================] - 0s 318us/step - loss: 0.4735 - KL_loss: 0.1372 - recon_loss: 0.3363 - val_loss: 0.4424 - val_KL_loss: 0.1337 - val_recon_loss: 0.3087
    Epoch 77/100
    400/400 [==============================] - 0s 325us/step - loss: 0.4506 - KL_loss: 0.1308 - recon_loss: 0.3198 - val_loss: 0.4277 - val_KL_loss: 0.1273 - val_recon_loss: 0.3004
    Epoch 78/100
    400/400 [==============================] - 0s 293us/step - loss: 0.4394 - KL_loss: 0.1244 - recon_loss: 0.3150 - val_loss: 0.4133 - val_KL_loss: 0.1209 - val_recon_loss: 0.2925
    Epoch 79/100
    400/400 [==============================] - 0s 423us/step - loss: 0.4266 - KL_loss: 0.1178 - recon_loss: 0.3089 - val_loss: 0.3992 - val_KL_loss: 0.1142 - val_recon_loss: 0.2850
    Epoch 80/100
    400/400 [==============================] - 0s 408us/step - loss: 0.4097 - KL_loss: 0.1111 - recon_loss: 0.2986 - val_loss: 0.3854 - val_KL_loss: 0.1074 - val_recon_loss: 0.2779
    Epoch 81/100
    400/400 [==============================] - 0s 364us/step - loss: 0.3991 - KL_loss: 0.1045 - recon_loss: 0.2947 - val_loss: 0.3719 - val_KL_loss: 0.1010 - val_recon_loss: 0.2709
    Epoch 82/100
    400/400 [==============================] - 0s 278us/step - loss: 0.3892 - KL_loss: 0.0981 - recon_loss: 0.2911 - val_loss: 0.3589 - val_KL_loss: 0.0948 - val_recon_loss: 0.2641
    Epoch 83/100
    400/400 [==============================] - 0s 258us/step - loss: 0.3723 - KL_loss: 0.0921 - recon_loss: 0.2802 - val_loss: 0.3463 - val_KL_loss: 0.0890 - val_recon_loss: 0.2573
    Epoch 84/100
    400/400 [==============================] - 0s 260us/step - loss: 0.3623 - KL_loss: 0.0865 - recon_loss: 0.2758 - val_loss: 0.3341 - val_KL_loss: 0.0836 - val_recon_loss: 0.2504
    Epoch 85/100
    400/400 [==============================] - 0s 260us/step - loss: 0.3493 - KL_loss: 0.0813 - recon_loss: 0.2680 - val_loss: 0.3223 - val_KL_loss: 0.0787 - val_recon_loss: 0.2436
    Epoch 86/100
    400/400 [==============================] - 0s 255us/step - loss: 0.3345 - KL_loss: 0.0766 - recon_loss: 0.2579 - val_loss: 0.3110 - val_KL_loss: 0.0741 - val_recon_loss: 0.2369
    Epoch 87/100
    400/400 [==============================] - 0s 245us/step - loss: 0.3266 - KL_loss: 0.0721 - recon_loss: 0.2544 - val_loss: 0.3001 - val_KL_loss: 0.0698 - val_recon_loss: 0.2303
    Epoch 88/100
    400/400 [==============================] - 0s 275us/step - loss: 0.3148 - KL_loss: 0.0680 - recon_loss: 0.2468 - val_loss: 0.2896 - val_KL_loss: 0.0658 - val_recon_loss: 0.2238
    Epoch 89/100
    400/400 [==============================] - 0s 298us/step - loss: 0.3029 - KL_loss: 0.0641 - recon_loss: 0.2388 - val_loss: 0.2795 - val_KL_loss: 0.0620 - val_recon_loss: 0.2175
    Epoch 90/100
    400/400 [==============================] - 0s 295us/step - loss: 0.2977 - KL_loss: 0.0604 - recon_loss: 0.2373 - val_loss: 0.2698 - val_KL_loss: 0.0585 - val_recon_loss: 0.2113
    Epoch 91/100
    400/400 [==============================] - 0s 345us/step - loss: 0.2885 - KL_loss: 0.0569 - recon_loss: 0.2316 - val_loss: 0.2604 - val_KL_loss: 0.0551 - val_recon_loss: 0.2052
    Epoch 92/100
    400/400 [==============================] - 0s 330us/step - loss: 0.2727 - KL_loss: 0.0537 - recon_loss: 0.2190 - val_loss: 0.2513 - val_KL_loss: 0.0520 - val_recon_loss: 0.1994
    Epoch 93/100
    400/400 [==============================] - 0s 228us/step - loss: 0.2707 - KL_loss: 0.0505 - recon_loss: 0.2202 - val_loss: 0.2426 - val_KL_loss: 0.0489 - val_recon_loss: 0.1937
    Epoch 94/100
    400/400 [==============================] - 0s 183us/step - loss: 0.2578 - KL_loss: 0.0476 - recon_loss: 0.2103 - val_loss: 0.2343 - val_KL_loss: 0.0460 - val_recon_loss: 0.1882
    Epoch 95/100
    400/400 [==============================] - 0s 168us/step - loss: 0.2479 - KL_loss: 0.0447 - recon_loss: 0.2031 - val_loss: 0.2262 - val_KL_loss: 0.0433 - val_recon_loss: 0.1829
    Epoch 96/100
    400/400 [==============================] - 0s 178us/step - loss: 0.2447 - KL_loss: 0.0421 - recon_loss: 0.2026 - val_loss: 0.2184 - val_KL_loss: 0.0407 - val_recon_loss: 0.1777
    Epoch 97/100
    400/400 [==============================] - 0s 170us/step - loss: 0.2322 - KL_loss: 0.0396 - recon_loss: 0.1926 - val_loss: 0.2110 - val_KL_loss: 0.0383 - val_recon_loss: 0.1727
    Epoch 98/100
    400/400 [==============================] - 0s 165us/step - loss: 0.2289 - KL_loss: 0.0372 - recon_loss: 0.1918 - val_loss: 0.2038 - val_KL_loss: 0.0359 - val_recon_loss: 0.1679
    Epoch 99/100
    400/400 [==============================] - 0s 173us/step - loss: 0.2240 - KL_loss: 0.0349 - recon_loss: 0.1891 - val_loss: 0.1969 - val_KL_loss: 0.0337 - val_recon_loss: 0.1632
    Epoch 100/100
    400/400 [==============================] - 0s 188us/step - loss: 0.2156 - KL_loss: 0.0328 - recon_loss: 0.1828 - val_loss: 0.1903 - val_KL_loss: 0.0317 - val_recon_loss: 0.1586
    
------------------  Plot the loss


```python
# plots - Loss
plt.plot(cvae_hist.history['loss'])
plt.plot(cvae_hist.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
```


![png](/images/Blog_CVAE_CREDIT_FRAUD_files/output_69_0.png)


## Reconstrution of new fraud data
    - function to create input vector that will be passed to decoder
    - output of decoder will be the desired generated data

```python
def construct_numvec(digit, z = None):
    out = np.zeros((1, n_z + n_y))
    
    out[:,  n_z] = digit
    #print(out)
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:,i] = z[i]
        #print('out',out)
        return(out)
```

#### Generate 10000 data and store it to generated_data list

```python
generated_data = []
```

```python
dig   = 1
sides = 10000
max_z = 1.5

img_it = 0

for i in range(0, sides):
    z1 = (((i / (sides-1)) * max_z)*2) - max_z
    z_ = [z1]
    for j in range(i, i+1):
        z2 = (((j / (sides-1)) * max_z)*2) - max_z
        z_.append(z2)

        
    vec     = construct_numvec(1, z_)
    decoded = decoder.predict(vec)
    generated_data.append(decoded)
        
```

```python
len(generated_data)
```




    10000



```python
generated_data[1]
```




    array([[9.9944490e-01, 8.6101500e-05, 8.6668078e-05, 1.0613600e-03,
            3.9377506e-04, 1.8465052e-04, 5.3125998e-04, 6.1531906e-04,
            1.6161945e-04, 4.2944402e-04, 2.1095443e-04, 3.6066069e-04,
            4.0693159e-04, 3.8716337e-04, 1.2618023e-03, 1.0135764e-03,
            4.6511332e-04, 6.2088552e-04, 4.7179303e-04, 2.5567663e-04,
            4.8512296e-04, 3.6986877e-04, 5.9142745e-05, 5.1477528e-04,
            2.5388756e-04, 6.5668876e-04, 5.8887445e-04, 2.6733367e-04,
            2.0034057e-03, 4.0071347e-04]], dtype=float32)



```python
# ------------------- Convert list to arry

generated_data = np.array(generated_data)
```

```python
generated_data = generated_data.reshape(10000,30)
generated_data.shape
```




    (10000, 30)



- Data generated have 30 features 
- We will add label data column to it

```python
generated_data = np.concatenate((generated_data, np.array([1]*10000).reshape(len(generated_data),1)), axis =1)
```

```python
generated_data.shape
```




    (10000, 31)



- concat generated data to Original data

```python
all_inp = np.concatenate((np.array(data),generated_data), axis = 0)
```

```python
all_inp.shape
```




    (294807, 31)



```python
inp_df = pd.DataFrame(all_inp)
```

```python
inp_df.columns = columns
```

```python
inp_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 31 columns</p>
</div>



- check for the combined input shape

```python
inp_df.Class.value_counts()
```




    0.0    284315
    1.0     10492
    Name: Class, dtype: int64



#### Apply Logistic Regression for modelling

```python
from sklearn.linear_model import LogisticRegression
```

```python
clf = LogisticRegression()
```

```python
from sklearn.model_selection import train_test_split
```

```python
train_xx, test_xx, train_y, test_y = train_test_split(inp_df.drop("Class", axis =1),  inp_df.Class)
```

```python
train_xx.shape, test_xx.shape, train_y.shape
```




    ((221105, 30), (73702, 30), (221105,))



```python
clf.fit(train_xx, train_y)
```




    LogisticRegression()



```python
pred = clf.predict(test_xx)
```

```python
(pred == test_y).value_counts()
```




    True     73273
    False      429
    Name: Class, dtype: int64



#### Calculate accuracy matrics

```python
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,accuracy_score
```

```python

print("accuracy", accuracy_score(test_y, pred))
print("precision", precision_score(test_y, pred))
print("recall", recall_score(test_y, pred))
print("f1", f1_score(test_y, pred))
print("confusion", confusion_matrix(test_y, pred))
```

    accuracy 0.994179262435212
    precision 0.8663911845730028
    recall 0.9839655846695347
    f1 0.9214429591649882
    confusion [[70757   388]
     [   41  2516]]
    

## Recall is 98.19 %
## F1 is 92% 
## It is far better than previous best


# Conclusion 

There are many ways to do fraud detection. But generative way solves problem of data skewness and provides far better
accuracies. 
### Please do suggest for improvement areas
### Do post for any queries
