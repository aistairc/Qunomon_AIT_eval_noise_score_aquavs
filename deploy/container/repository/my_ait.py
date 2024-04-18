#!/usr/bin/env python
# coding: utf-8

# # AIT Development notebook

# ## notebook of structure

# | #  | Name                                               | cells | for_dev | edit               | description                                                                |
# |----|----------------------------------------------------|-------|---------|--------------------|----------------------------------------------------------------------------|
# | 1  | [Environment detection](##1-Environment-detection) | 1     | No      | uneditable         | detect whether the notebook are invoked for packaging or in production     |
# | 2  | [Preparing AIT SDK](##2-Preparing-AIT-SDK)         | 1     | Yes     | uneditable         | download and install AIT SDK                                               |
# | 3  | [Dependency Management](##3-Dependency-Management) | 3     | Yes     | required(cell #2)  | generate requirements.txt for Docker container                             |
# | 4  | [Importing Libraries](##4-Importing-Libraries)     | 2     | Yes     | required(cell #1)  | import required libraries                                                  |
# | 5  | [Manifest Generation](##5-Manifest-Generation)     | 1     | Yes     | required           | generate AIT Manifest                                                      |
# | 6  | [Prepare for the Input](##6-Prepare-for-the-Input) | 1     | Yes     | required           | generate AIT Input JSON (inventory mapper)                                 |
# | 7  | [Initialization](##7-Initialization)               | 1     | No      | uneditable         | initialization for AIT execution                                           |
# | 8  | [Function definitions](##8-Function-definitions)   | N     | No      | required           | define functions invoked from Main area.<br> also define output functions. |
# | 9  | [Main Algorithms](##9-Main-Algorithms)             | 1     | No      | required           | area for main algorithms of an AIT                                         |
# | 10 | [Entry point](##10-Entry-point)                    | 1     | No      | uneditable         | an entry point where Qunomon invoke this AIT from here                     |
# | 11 | [License](##11-License)                            | 1     | Yes     | required           | generate license information                                               |
# | 12 | [Deployment](##12-Deployment)                      | 1     | Yes     | uneditable         | convert this notebook to the python file for packaging purpose             |

# ## notebook template revision history

# 1.0.1 2020/10/21
# 
# * add revision history
# * separate `create requirements and pip install` editable and noeditable
# * separate `import` editable and noeditable
# 
# 1.0.0 2020/10/12
# 
# * new cerarion

# ## body

# ### #1 Environment detection

# [uneditable]

# In[1]:


# Determine whether to start AIT or jupyter by startup argument
import sys
is_ait_launch = (len(sys.argv) == 2)


# ### #2 Preparing AIT SDK

# [uneditable]

# In[2]:


if not is_ait_launch:
    # get ait-sdk file name
    from pathlib import Path
    from glob import glob
    import re
    import os

    current_dir = get_ipython().run_line_magic('pwd', '')

    ait_sdk_path = "./ait_sdk-*-py3-none-any.whl"
    ait_sdk_list = glob(ait_sdk_path)
    ait_sdk_name = os.path.basename(ait_sdk_list[-1])

    # install ait-sdk
    get_ipython().system('pip install -q --upgrade pip')
    get_ipython().system('pip install -q --no-deps --force-reinstall ./$ait_sdk_name')


# ### #3 Dependency Management

# #### #3-1 [uneditable]

# In[3]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_requirements_generator import AITRequirementsGenerator
    requirements_generator = AITRequirementsGenerator()


# #### #3-2 [required]

# In[4]:


if not is_ait_launch:
    requirements_generator.add_package('numpy', '1.26.3')
    requirements_generator.add_package('matplotlib', '3.7.3')
    requirements_generator.add_package('pandas', '2.2.2')
    requirements_generator.add_package('scikit-learn', '1.4.2')
    requirements_generator.add_package('tensorflow', '2.11.1')
    requirements_generator.add_package('scipy', '1.13.0')


# #### #3-3 [uneditable]

# In[5]:


if not is_ait_launch:
    requirements_generator.add_package(f'./{ait_sdk_name}')
    requirements_path = requirements_generator.create_requirements(current_dir)

    get_ipython().system('pip install -q -r $requirements_path ')


# ### #4 Importing Libraries

# #### #4-1 [required]

# In[6]:


# import if you need modules cell
# from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape, BatchNormalization, MaxPooling2D, Dropout
from tensorflow.keras import backend as K
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from scipy.stats import norm

from collections import defaultdict 
import random
import numpy as np
from sklearn.utils import shuffle
from scipy import stats
from sklearn.metrics import precision_score, recall_score, accuracy_score 
from collections import Counter

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef


# #### #4-2 [uneditable]

# In[7]:


# must use modules
from os import path
import shutil  # do not remove
from ait_sdk.common.files.ait_input import AITInput  # do not remove
from ait_sdk.common.files.ait_output import AITOutput  # do not remove
from ait_sdk.common.files.ait_manifest import AITManifest  # do not remove
from ait_sdk.develop.ait_path_helper import AITPathHelper  # do not remove
from ait_sdk.utils.logging import get_logger, log, get_log_path  # do not remove
from ait_sdk.develop.annotation import measures, resources, downloads, ait_main  # do not remove
# must use modules


# ### #5 Manifest Generation

# [required]

# In[8]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_manifest_generator import AITManifestGenerator
    manifest_generator = AITManifestGenerator(current_dir)
    manifest_generator.set_ait_name('eval_noise_score_aquavs')
    manifest_generator.set_ait_description('モデルの安定性を評価するために、ノイズを付けたラベルで検証します。SVAEの潜在表現を使用し、入力データセット内の各サンプルの異常を測定する「ノイズスコア」を計測します。詳細については、元の論文「Pulastya, et al. Assessing the quality of the datasets by identifying mislabeled samples」(URL: https://dl.acm.org/doi/abs/10.1145/3487351.3488361)')

    manifest_generator.set_ait_source_repository('https://github.com/aistairc/Qunomon_AIT_eval_noise_score_aquavs')
    manifest_generator.set_ait_version('1.0')
    manifest_generator.add_ait_keywords('Evaluation')
    manifest_generator.set_ait_quality('https://ait-hub.pj.aist.go.jp/ait-hub/api/0.0.1/qualityDimensions/機械学習品質マネジメントガイドライン第三版/A-1問題領域分析の十分性')
    
    inventory_requirement_mnist_data         = manifest_generator.format_ait_inventory_requirement(format_=['npz'])
    inventory_requirement_fashion_mnist_data = manifest_generator.format_ait_inventory_requirement(format_=['npz'])
    inventory_requirement_cifar10_data       = manifest_generator.format_ait_inventory_requirement(format_=['npz'])
    inventory_requirement_cifar100_data      = manifest_generator.format_ait_inventory_requirement(format_=['npz'])

    manifest_generator.add_ait_inventories(name='mnist_data', type_='dataset', description='mnist data', requirement=inventory_requirement_mnist_data)
    manifest_generator.add_ait_inventories(name='fashion_mnist_data', type_='dataset', description='fashion mnist data', requirement=inventory_requirement_fashion_mnist_data)
    manifest_generator.add_ait_inventories(name='cifar10_data', type_='dataset', description='cifar10 data', requirement=inventory_requirement_cifar10_data)
    manifest_generator.add_ait_inventories(name='cifar100_data', type_='dataset', description='cifar100 data', requirement=inventory_requirement_cifar100_data)

    ### input parameters, Hyperparameters
    manifest_generator.add_ait_parameters(name='MAD_Outlier_constant', type_='float', default_val='1.5', description='Hyperparameter specifying the outlier detection in latent space')
    manifest_generator.add_ait_parameters(name='MISLABEL_THRESHOLD', type_='float', default_val='0.5', description='Hyperparameter specifying the mislabel based on fraction of outlier dimensions')
    manifest_generator.add_ait_parameters(name='latent_dim', type_='int', default_val='100', description='Hyperparameter specifying the latent space dimension')
    manifest_generator.add_ait_parameters(name='batch_size', type_='int', default_val='32', description='Hyperparameter specifying the batch size of the optimizer of VAE')
    
    ### input parameters
    manifest_generator.add_ait_parameters(name='datasetName', type_='str', default_val='mnist', description='Parameter specifying dataset')
    manifest_generator.add_ait_parameters(name='noise_perc', type_='float', default_val='20', description='Parameter specifying the percentage of noised labels')
    manifest_generator.add_ait_parameters(name='noise_systematic', type_='str', default_val='Sys', description='Parameter specifying the type to add noise according to the label values (Sys) or random (Uni)')
    manifest_generator.add_ait_parameters(name='model_name', type_='str', default_val='', description='Parameter specifying VAE model')
    
    ### measures: evaluation metrics
    manifest_generator.add_ait_measures(name='evaluation_result_accuracy', type_='float', structure='single', min='0', max='1', description='accuracy')
    manifest_generator.add_ait_measures(name='evaluation_result_precision', type_='float', structure='single', min='0', max='1', description='precision')
    manifest_generator.add_ait_measures(name='evaluation_result_recall', type_='float', structure='single', min='0', max='1', description='recall')
    manifest_generator.add_ait_measures(name='evaluation_result_f1', type_='float', structure='single', min='0', max='1', description='f1')
    manifest_generator.add_ait_measures(name='evaluation_result_roc_auc', type_='float', structure='single', min='0', max='1', description='roc_auc')
    manifest_generator.add_ait_measures(name='evaluation_result_mcc', type_='float', structure='single', min='0', max='1', description='mcc')

    ### download: VAE model
    manifest_generator.add_ait_downloads(name='vae', description='VAE model learned')
    manifest_generator.add_ait_downloads(name='log', description='AIT execution logs')
    manifest_path = manifest_generator.write()


# ### #6 Prepare for the Input

# [required]

# In[9]:


if not is_ait_launch:
    from ait_sdk.common.files.ait_input_generator import AITInputGenerator
    input_generator = AITInputGenerator(manifest_path)
    input_generator.add_ait_inventories(name='mnist_data',
                                        value='mnist_data/mnist_train_data.npz')
    input_generator.add_ait_inventories(name='fashion_mnist_data',
                                        value='fashion_mnist_data/fashion_mnist_train_data.npz')
    input_generator.add_ait_inventories(name='cifar10_data',
                                        value='cifar10_data/cifar10_train_data.npz')
    input_generator.add_ait_inventories(name='cifar100_data',
                                        value='cifar100_data/cifar100_train_data.npz')
    
    ### hyperparameter
    MAD_Outlier_constant = 1.5
    MISLABEL_THRESHOLD = 0.5
    latent_dim = 100
    batch_size = 32
    
    input_generator.set_ait_params(name='MAD_Outlier_constant', value=MAD_Outlier_constant)
    input_generator.set_ait_params(name='MISLABEL_THRESHOLD', value=MISLABEL_THRESHOLD)
    input_generator.set_ait_params(name='latent_dim', value=latent_dim)
    input_generator.set_ait_params(name='batch_size', value=batch_size)
    
    ### input parameters
    datasetName = 'mnist'
    noise_perc = 10
    noise_systematic = 'Sys'
    model_name = f'vae_{datasetName}_{noise_systematic}_{noise_perc}.keras'
    
    input_generator.set_ait_params(name='datasetName', value=datasetName)
    input_generator.set_ait_params(name='noise_perc', value=noise_perc)
    input_generator.set_ait_params(name='noise_systematic', value=noise_systematic)
    input_generator.set_ait_params(name='model_name', value=model_name)
        
    input_generator.write()


# ### #7 Initialization

# [uneditable]

# In[10]:


logger = get_logger()

ait_manifest = AITManifest()
ait_input = AITInput(ait_manifest)
ait_output = AITOutput(ait_manifest)

if is_ait_launch:
    # launch from AIT
    current_dir = path.dirname(path.abspath(__file__))
    path_helper = AITPathHelper(argv=sys.argv, ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)
else:
    # launch from jupyter notebook
    # ait.input.json make in input_dir
    input_dir = '/usr/local/qai/mnt/ip/job_args/1/1'
    current_dir = get_ipython().run_line_magic('pwd', '')
    path_helper = AITPathHelper(argv=['', input_dir], ait_input=ait_input, ait_manifest=ait_manifest, entry_point_dir=current_dir)

ait_input.read_json(path_helper.get_input_file_path())
ait_manifest.read_json(path_helper.get_manifest_file_path())

### do not edit cell


# ### #8 Function definitions

# [required]

# In[11]:


### helper function
def specify_dimensions(datasetName):
    if datasetName == 'mnist' or datasetName == 'fashion_mnist':
        # constants to specify model details
        img_dimensions = (28, 28, 1)
        num_classes = 10
        num_channels = 1
    elif datasetName == 'cifar10':
        # constants to specify model details
        img_dimensions = (32, 32, 3)
        num_classes = 10
        num_channels = 3
    elif datasetName == 'cifar100':
        # constants to specify model details
        img_dimensions = (32, 32, 3)
        num_classes = 100
        num_channels = 3
    else:
        raise ValueError
    return img_dimensions, num_classes, num_channels

def prepare_data(datasetName, train_data, train_labels, noisePerc, noiseType):
    img_dimensions, num_classes, num_channels = specify_dimensions(datasetName)
    
    #adds noise to y-labels using uniform noise model - i.e. mislabeled samples are given labels uniformly at random.
    def add_noise_UniformNoiseModel(input_y, perc, allClasses):
        final_idx = defaultdict(list)
        noisy_y = [-1 for i in range(input_y.shape[0])]

        for i in range(input_y.shape[0]): 
            final_idx[input_y[i]].append(i)

        for lbl in final_idx.keys():
            remC = (perc/100.0)*len(final_idx[lbl])
            #print("Label: ", lbl, "; # of datapoints flipped: ", int(remC))
            for i in range(int(remC)):
                idx = random.randint(0, len(final_idx[lbl]) - 1)
                newLabel = random.choice(allClasses)
                while (newLabel == lbl):
                    newLabel = random.choice(allClasses)
                noisy_y[final_idx[lbl][idx]] = newLabel  # update the label for datapoint from `label` to `newLabel` 
                del final_idx[lbl][idx]

        for lbl in final_idx.keys():
            for i in final_idx[lbl]:
                noisy_y[i] = lbl

        return np.array(noisy_y)

    #adds noise to y-labels using systematic noise model - i.e. mislabeled samples are given labels systematic at random.
    def add_noise_SystematicNoiseModel(input_y, perc, allClasses):
        final_idx = defaultdict(list)
        noisy_y = [-1 for i in range(input_y.shape[0])]

        for i in range(input_y.shape[0]): 
            final_idx[input_y[i]].append(i)

        for lbl in final_idx.keys():
            remC = (perc/100.0)*len(final_idx[lbl])
            #print("Label: ", lbl, "; # of datapoints flipped: ", int(remC))
            for i in range(int(remC)):
                idx = random.randint(0, len(final_idx[lbl]) - 1)
                newLabel = (lbl + 1)%(len(allClasses))
                noisy_y[final_idx[lbl][idx]] = newLabel  # update the label for datapoint from `label` to `newLabel` 
                del final_idx[lbl][idx]

        for lbl in final_idx.keys():
            for i in final_idx[lbl]:
                noisy_y[i] = lbl

        return np.array(noisy_y)

    # min-max normalization
    def min_max_normalize(lis):
        minL = float(min(lis))
        maxL = float(max(lis))
        minMaxLis = [float((float(x) - minL)/ (maxL - minL)) for x in lis]
        return minMaxLis  
    
    #reshaping
    train_data = train_data.reshape((train_data.shape[0], img_dimensions[0], img_dimensions[1], img_dimensions[2]))
    train_labels = train_labels.reshape(train_labels.shape[0])

    # convert from integers to floats
    train_data = train_data.astype('float32')

    # normalize to range 0-1
    train_data = train_data / 255.0

    if(noiseType == "Sys"):
        noisy_labels = add_noise_SystematicNoiseModel(train_labels, noisePerc, [cl for cl in range(10)])
    elif(noiseType == "Uni"):
        noisy_labels = add_noise_UniformNoiseModel(train_labels, noisePerc, [cl for cl in range(10)])

    grn_truth = np.array(noisy_labels == train_labels, dtype=int)

    print("Number of mislabeled: ", len(grn_truth) - sum(grn_truth), "out of", len(grn_truth))

    y_enc_noisy_labels = tf.keras.utils.to_categorical(noisy_labels) #encode noisy labels

    return train_data, noisy_labels, grn_truth, y_enc_noisy_labels

#grouping datapoints by respective classes
def group_data_by_class(input_x, input_y):
    final_out = defaultdict(list) 
    final_idx = defaultdict(list)
    for i in range(input_x.shape[0]): 
        final_out[input_y[i]].append(input_x[i])
        final_idx[input_y[i]].append(i)
    return final_out, final_idx

#Ref - https://core.ac.uk/download/pdf/206095228.pdf
def outlier_detection_med_mad(input_data, k1):
    column_med = np.median(input_data, axis = 0)
    column_mad = stats.median_abs_deviation(input_data,axis = 0)

    #computing threshold for each feature
    threshold_lower = column_med - (k1*column_mad)
    threshold_upper = column_med + (k1*column_mad)
    outliers = []
    num_outlier_feature_list = []
    outlier_level = defaultdict(list)
    for i in range(input_data.shape[0]):
        num_outlier_feature = 0
        x = input_data[i]
        for id in range(x.shape[0]):
            if not (threshold_lower[id] <= x[id] and x[id] <= threshold_upper[id]):
                num_outlier_feature += 1
        outlier_level[num_outlier_feature].append(i)
    return outlier_level

# computes noise level of each datapoint 
def get_train_lvl(encoder, input_x, input_y, MAD_Outlier_constant):
    grouped_train, grouped_idx = group_data_by_class(input_x, input_y.reshape(input_y.shape[0]))
    cntr = 0
    train_lvl = [-1 for i in range(input_x.shape[0])]
    for digit in range(0,10):
        z_values = encoder.predict(np.array(grouped_train[digit]))[2]
        class_outliers = outlier_detection_med_mad(z_values, MAD_Outlier_constant)
        for i in class_outliers.keys():
            for j in class_outliers[i]:
                # i is the outlier level
                # grouped_idx[digit][j] is the index
                train_lvl[grouped_idx[digit][j]] = i
    return np.array(train_lvl)

### main functions
@log(logger)
@downloads(ait_output, path_helper, 'vae', model_name)
def optimize_AQUAVS(datasetName, train_data, y_enc_noisy_labels, file_path = None):
    img_dimensions, num_classes, num_channels = specify_dimensions(datasetName)
    
    # VAE 
    def vae_loss(data, reconstruction):
        z_mean, z_log_var, z = encoder(data)
        reconstruction_loss = keras.losses.binary_crossentropy(data, reconstruction)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss, axis=[1,2])
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss, axis=1)
        kl_loss *= -0.5
        total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)/100
        return total_loss

    def sampling(args):
        z_mean, z_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0,)  ## latent_dim = K.shape(z_mean)[1] 
        return z_mean + K.exp(z_var / 2) * epsilon

    ## ENCODER
    inputNode = Input(shape=img_dimensions, name="EncoderInput")
    enc_inter = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', kernel_initializer='he_uniform')(inputNode)
    enc_inter = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', kernel_initializer='he_uniform', activation='relu')(enc_inter)
    enc_inter = Conv2D(filters=128, kernel_size=4, strides=1, padding='same', kernel_initializer='he_uniform', activation=tf.nn.relu)(enc_inter)

    conv_shape = K.int_shape(enc_inter) 

    enc_inter = Flatten()(enc_inter)
    z_mean = Dense(latent_dim, name="Mean")(enc_inter)
    z_var = Dense(latent_dim, name="Variance")(enc_inter)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_var])
    encoder = Model(inputNode, [z_mean, z_var, z], name="Encoder")

    ## CLASSIFIER
    clf_latent_inputs = Input(shape=(latent_dim,), name='ClassifierInput')
    clf_outputs = Dense(num_classes, activation='softmax', name='ClassifierOutput')(clf_latent_inputs)
    clf_supervised = Model(clf_latent_inputs, clf_outputs, name='Classifier')

    ## DECODER
    inputNode2 = Input(shape=(latent_dim,), name="DecoderInput")
    dec_inter = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3])(inputNode2)
    dec_inter = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(dec_inter)
    dec_inter = Conv2DTranspose(filters=128, kernel_size=4, strides=1, padding='same', kernel_initializer='he_uniform', activation='relu')(dec_inter)
    dec_inter = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', kernel_initializer='he_uniform', activation='relu')(dec_inter)
    dec_inter = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', kernel_initializer='he_uniform', activation='relu')(dec_inter)
    decoder_node = Conv2DTranspose(num_channels, kernel_size=4, strides=1, padding='same')(dec_inter)
    decoder = Model(inputNode2, decoder_node, name='Decoder')

    output_combined = [decoder(encoder(inputNode)[2]), clf_supervised(encoder(inputNode)[2])]
    vae = Model(inputNode, output_combined, name='S-VAE')

    vae.compile(optimizer='adam', loss=[vae_loss, 'categorical_crossentropy'])

    # callback definitions
    def scheduler(epoch):
        return 0.001/(epoch+1)

    earlyStopCallback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    lrScheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

    splitID = int(0.8*len(train_data))

    #Note - VAE trains on noisy data
    vae.fit(train_data[:splitID], [train_data[:splitID], y_enc_noisy_labels[:splitID]], 
            shuffle=True, epochs=10, batch_size=32, 
            validation_data=(train_data[splitID:], [train_data[splitID:], y_enc_noisy_labels[splitID:]]), 
            callbacks=[lrScheduler, earlyStopCallback],
            verbose=1)

    model_name = file_path
    if model_name is not None:
        vae.save(model_name)
    return vae


@log(logger)
@measures(ait_output, 'evaluation_result_accuracy')
def evaluate_accuracy_AQUAVS(encoder, train_data, noisy_labels, grn_truth):
    
    noisy_lvl = get_train_lvl(encoder, train_data, noisy_labels, MAD_Outlier_constant)

    # identify mislabel
    true_mislabels = grn_truth
    estimated_mislabels = np.where(noisy_lvl / latent_dim <= MISLABEL_THRESHOLD, 1, 0)

    # evaluate the differnce between true_mislabels vs estimated_mislabels
    return accuracy_score(true_mislabels, estimated_mislabels)

@log(logger)
@measures(ait_output, 'evaluation_result_precision')
def evaluate_precision_AQUAVS(encoder, train_data, noisy_labels, grn_truth):
    
    noisy_lvl = get_train_lvl(encoder, train_data, noisy_labels, MAD_Outlier_constant)

    # identify mislabel
    true_mislabels = grn_truth
    estimated_mislabels = np.where(noisy_lvl / latent_dim <= MISLABEL_THRESHOLD, 1, 0)

    # evaluate the differnce between true_mislabels vs estimated_mislabels
    return precision_score(true_mislabels, estimated_mislabels)

@log(logger)
@measures(ait_output, 'evaluation_result_recall')
def evaluate_recall_AQUAVS(encoder, train_data, noisy_labels, grn_truth):
    
    noisy_lvl = get_train_lvl(encoder, train_data, noisy_labels, MAD_Outlier_constant)

    # identify mislabel
    true_mislabels = grn_truth
    estimated_mislabels = np.where(noisy_lvl / latent_dim <= MISLABEL_THRESHOLD, 1, 0)

    # evaluate the differnce between true_mislabels vs estimated_mislabels
    return recall_score(true_mislabels, estimated_mislabels)

@log(logger)
@measures(ait_output, 'evaluation_result_f1')
def evaluate_f1_AQUAVS(encoder, train_data, noisy_labels, grn_truth):
    
    noisy_lvl = get_train_lvl(encoder, train_data, noisy_labels, MAD_Outlier_constant)

    # identify mislabel
    true_mislabels = grn_truth
    estimated_mislabels = np.where(noisy_lvl / latent_dim <= MISLABEL_THRESHOLD, 1, 0)

    # evaluate the differnce between true_mislabels vs estimated_mislabels
    return f1_score(true_mislabels, estimated_mislabels)

@log(logger)
@measures(ait_output, 'evaluation_result_roc_auc')
def evaluate_roc_auc_AQUAVS(encoder, train_data, noisy_labels, grn_truth):
    
    noisy_lvl = get_train_lvl(encoder, train_data, noisy_labels, MAD_Outlier_constant)

    # identify mislabel
    true_mislabels = grn_truth
    estimated_mislabels = np.where(noisy_lvl / latent_dim <= MISLABEL_THRESHOLD, 1, 0)

    # evaluate the differnce between true_mislabels vs estimated_mislabels
    return roc_auc_score(true_mislabels, estimated_mislabels)

@log(logger)
@measures(ait_output, 'evaluation_result_mcc')
def evaluate_mcc_AQUAVS(encoder, train_data, noisy_labels, grn_truth):
    
    noisy_lvl = get_train_lvl(encoder, train_data, noisy_labels, MAD_Outlier_constant)

    # identify mislabel
    true_mislabels = grn_truth
    estimated_mislabels = np.where(noisy_lvl / latent_dim <= MISLABEL_THRESHOLD, 1, 0)

    # evaluate the differnce between true_mislabels vs estimated_mislabels
    return matthews_corrcoef(true_mislabels, estimated_mislabels)


# In[12]:


@log(logger)
@downloads(ait_output, path_helper, 'Log', 'ait.log')
def move_log(file_path: str=None) -> str:
    shutil.move(get_log_path(), file_path)


# ### #9 Main Algorithms

# [required]

# In[13]:


@log(logger)
@ait_main(ait_output, path_helper)
def main() -> None:
    
    input_data = np.load(ait_input.get_inventory_path(f'{datasetName}_data'))
    train_data, noisy_labels, grn_truth, y_enc_noisy_labels = prepare_data(datasetName, input_data['X'], input_data['y'], noise_perc, noise_systematic)
    
    vae = optimize_AQUAVS(datasetName, train_data, y_enc_noisy_labels)
    encoder = vae.get_layer('Encoder')
    
    evaluation_result = evaluate_accuracy_AQUAVS(encoder, train_data, noisy_labels, grn_truth)
    evaluation_result = evaluate_precision_AQUAVS(encoder, train_data, noisy_labels, grn_truth)
    evaluation_result = evaluate_recall_AQUAVS(encoder, train_data, noisy_labels, grn_truth)
    evaluation_result = evaluate_f1_AQUAVS(encoder, train_data, noisy_labels, grn_truth)
    evaluation_result = evaluate_roc_auc_AQUAVS(encoder, train_data, noisy_labels, grn_truth)
    evaluation_result = evaluate_mcc_AQUAVS(encoder, train_data, noisy_labels, grn_truth)
    
    move_log()


# ### #10 Entry point

# [uneditable]

# In[ ]:


if __name__ == '__main__':
    main()


# ### #11 License

# [required]

# In[ ]:


ait_owner='AIST'
ait_creation_year='2024'


# ### #12 Deployment

# [uneditable] 

# In[ ]:


if not is_ait_launch:
    from ait_sdk.deploy import prepare_deploy
    from ait_sdk.license.license_generator import LicenseGenerator
    
    current_dir = get_ipython().run_line_magic('pwd', '')
    prepare_deploy(ait_sdk_name, current_dir, requirements_path)
    
    # output License.txt
    license_generator = LicenseGenerator()
    license_generator.write('../top_dir/LICENSE.txt', ait_creation_year, ait_owner)


# In[ ]:




