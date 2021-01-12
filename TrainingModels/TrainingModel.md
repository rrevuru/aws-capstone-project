# Importing Libraries


```python
import pandas as pd
import numpy as np
import sagemaker
import boto3, csv, io, json
from sklearn.model_selection import train_test_split
import sagemaker.amazon.common as smac
from sagemaker import get_execution_role
from sagemaker.predictor import json_deserializer
from scipy.sparse import lil_matrix
from sagemaker.amazon.amazon_estimator import get_image_uri
sage_client = boto3.Session().client('sagemaker')
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
```

# Downloading data from S3 bucket


```python
!aws s3 cp s3://movies-mlready-bucket/run-1607268992801-part-r-00000.csv .
```

    download: s3://movies-mlready-bucket/run-1607268992801-part-r-00000.csv to ./run-1607268992801-part-r-00000.csv


# Reading the CSV file into dataframe, and converting date type to integer


```python
df = pd.read_csv('run-1607268992801-part-r-00000.csv')
```


```python
df = df.drop(df[df.userid == 'userId'].index)
```


```python
df.userid = df.userid.astype(int)
```


```python
df.movieid = df.movieid.astype(int)
```


```python
df.rating = df.rating.astype(float)
```


```python
df_new = df.drop(df[df['movieid']>10000].index)
```


```python
df_new.shape
```




    (77939, 4)



# Splitting the data into Train & Test 80/20


```python
train, test = train_test_split(df_new, test_size=0.20)
```


```python
train.shape
```




    (62351, 4)




```python
test.shape
```




    (15588, 4)




```python
nb_users= train['userid'].max()
nb_movies=train['movieid'].max()
nb_features=nb_users+nb_movies
nb_ratings_train=len(train.index)
nb_ratings_test=len(test.index)
print (" # of users: ", nb_users)
print (" # of movies: ", nb_movies)
print (" Training Count: ", nb_ratings_train)
print (" Testing Count: ", nb_ratings_test)
print (" Features (# of users + # of movies): ", nb_features)
```

     # of users:  610
     # of movies:  9018
     Training Count:  62351
     Testing Count:  15588
     Features (# of users + # of movies):  9628


# Helper function to convert the dataframe to a sparse matrix with total of 9628 features (Courtesy Julie Simon)


```python
def loadDataset(df, lines, columns):
    # Features are one-hot encoded in a sparse matrix
    X = lil_matrix((lines, columns)).astype('float32')
    # Labels are stored in a vector
    Y = []
    line=0
    for index, row in df.iterrows():
            X[line,row['userid']-1] = 1
            X[line, nb_users+(row['movieid']-1)] = 1
            if int(row['rating']) >= 4:
                Y.append(1)
            else:
                Y.append(0)
            line=line+1

    Y=np.array(Y).astype('float32')            
    return X,Y


X_train, Y_train = loadDataset(train, nb_ratings_train, nb_features)
X_test, Y_test = loadDataset(test, nb_ratings_test, nb_features)
```


```python
print(X_train.shape)
print(Y_train.shape)
assert X_train.shape == (nb_ratings_train, nb_features)
assert Y_train.shape == (nb_ratings_train, )
zero_labels = np.count_nonzero(Y_train)
print("Training labels: %d zeros, %d ones" % (zero_labels, nb_ratings_train-zero_labels))

print(X_test.shape)
print(Y_test.shape)
assert X_test.shape  == (nb_ratings_test, nb_features)
assert Y_test.shape  == (nb_ratings_test, )
zero_labels = np.count_nonzero(Y_test)
print("Test labels: %d zeros, %d ones" % (zero_labels, nb_ratings_test-zero_labels))
```

    (62351, 9628)
    (62351,)
    Training labels: 30454 zeros, 31897 ones
    (15588, 9628)
    (15588,)
    Test labels: 7509 zeros, 8079 ones



```python
#Change this value to your own bucket name
bucket = 'movies-mlready-bucket'
prefix = 'fm'

train_key      = 'train.protobuf'
train_prefix   = '{}/{}'.format(prefix, 'train')

test_key       = 'test.protobuf'
test_prefix    = '{}/{}'.format(prefix, 'test')


output_prefix  = 's3://{}/{}/output'.format(bucket, prefix)
```

# Helper function to convert the training and test data into protobuf format(Courtesy Julie Simon)


```python
def writeDatasetToProtobuf(X, bucket, prefix, key, d_type, Y=None):
    buf = io.BytesIO()
    if d_type == "sparse":
        smac.write_spmatrix_to_sparse_tensor(buf, X, labels=Y)
    else:
        smac.write_numpy_to_dense_tensor(buf, X, labels=Y)
        
    buf.seek(0)
    obj = '{}/{}'.format(prefix, key)
    boto3.resource('s3').Bucket(bucket).Object(obj).upload_fileobj(buf)
    return 's3://{}/{}'.format(bucket,obj)
    
fm_train_data_path = writeDatasetToProtobuf(X_train, bucket, train_prefix, train_key, "sparse", Y_train)
fm_test_data_path  = writeDatasetToProtobuf(X_test, bucket, test_prefix, test_key, "sparse", Y_test)

  
print ("Training data S3 path: ",fm_train_data_path)
print ("Test data S3 path: ",fm_test_data_path)
print ("FM model output S3 path: {}".format(output_prefix))
```

    Training data S3 path:  s3://movies-mlready-bucket/fm/train/train.protobuf
    Test data S3 path:  s3://movies-mlready-bucket/fm/test/test.protobuf
    FM model output S3 path: s3://movies-mlready-bucket/fm/output


# Create a Factorization Machines estimator, with mandatory hyperparameters)


```python
instance_type='ml.m5.large'
fm = sagemaker.estimator.Estimator(get_image_uri(boto3.Session().region_name, "factorization-machines"),
                                   get_execution_role(), 
                                   train_instance_count=1, 
                                   train_instance_type=instance_type,
                                   base_job_name='V3Blackbelt',
                                   output_path=output_prefix,
                                   sagemaker_session=sagemaker.Session())

fm.set_hyperparameters(feature_dim=nb_features,
                      predictor_type='binary_classifier',
                      mini_batch_size=1000,
                      num_factors=64,
                      epochs=100)
```

    The method get_image_uri has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    Defaulting to the only supported framework/algorithm version: 1. Ignoring framework/algorithm version: 1.
    train_instance_count has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.
    train_instance_type has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.


# Hyperparameter Tuning Ranges


```python
hyperparameter_ranges = {'bias_init_sigma': ContinuousParameter(1e-8, 32),
                         'bias_lr': ContinuousParameter(1e-8,32),
                         'bias_wd': ContinuousParameter(1e-8,32),
                         'linear_init_sigma': ContinuousParameter(1e-8, 32),
                         'linear_lr': ContinuousParameter(1e-8,32),
                         'linear_wd': ContinuousParameter(1e-8,32),
                         'factors_init_sigma': ContinuousParameter(1e-8, 32),
                         'factors_lr': ContinuousParameter(1e-8,32),
                         'factors_wd': ContinuousParameter(1e-8,32)}
                    
```

# Objective Metric Name = Binary classification accuracy


```python
objective_metric_name = 'test:binary_classification_accuracy'
```

# Hyperparameter Tuner Initialization


```python
tuner = HyperparameterTuner(fm,
                            objective_metric_name,
                            hyperparameter_ranges,
                            base_tuning_job_name='V3Blackbelt',
                            max_jobs=5,
                            max_parallel_jobs=3)
```

# Fitting a Hyperparameter tuner


```python
tuner.fit({'train': fm_train_data_path, 'test': fm_test_data_path})
```

    ....................................................................................................................!



```python
fm_predictor = tuner.deploy(instance_type='ml.c4.xlarge', initial_instance_count=1)
```

    
    2020-12-15 15:12:53 Starting - Preparing the instances for training
    2020-12-15 15:12:53 Downloading - Downloading input data
    2020-12-15 15:12:53 Training - Training image download completed. Training in progress.
    2020-12-15 15:12:53 Uploading - Uploading generated training model
    2020-12-15 15:12:53 Completed - Training job completed
    ---------------!

# Inference serializer


```python
def fm_serializer(data):
    js = {"instances": []}
    for row in data:
        js["instances"].append({"features": row.tolist()})
    return json.dumps(js).encode()


fm_predictor.serializer.serialize = fm_serializer
fm_predictor.deserializer = json_deserializer
```

# Making Predictions


```python
result = fm_predictor.predict(X_test[1002:1009].toarray(), initial_args={"ContentType": "application/json"})
print(result)
```

    The json_deserializer has been renamed in sagemaker>=2.
    See: https://sagemaker.readthedocs.io/en/stable/v2.html for details.


    {'predictions': [{'score': 0.9999101161956787, 'predicted_label': 1.0}, {'score': 0.9998399019241333, 'predicted_label': 1.0}, {'score': 0.9996410608291626, 'predicted_label': 1.0}, {'score': 0.9999958276748657, 'predicted_label': 1.0}, {'score': 0.9999547004699707, 'predicted_label': 1.0}, {'score': 0.8490985631942749, 'predicted_label': 1.0}, {'score': 1.0, 'predicted_label': 1.0}]}


# Inference with Test data


```python
print (Y_test[1002:1009])
```

    [1. 1. 1. 1. 0. 1. 0.]



```python

```
