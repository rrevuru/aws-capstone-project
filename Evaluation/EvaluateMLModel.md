# Importing Libraries


```python
import pandas as pd
import numpy as np
import sagemaker, boto3
import sagemaker.amazon.common as smac
sage_client = boto3.Session().client('sagemaker')
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
```

# Getting the results from latest Hyperparameter Tuning Job


```python
tuning_job_name='V3Blackbelt-201213-2200'

# run this cell to check current status of hyperparameter tuning job
tuning_job_result = sage_client.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=tuning_job_name)

status = tuning_job_result['HyperParameterTuningJobStatus']
if status != 'Completed':
    print('Reminder: the tuning job has not been completed.')
    
job_count = tuning_job_result['TrainingJobStatusCounters']['Completed']
print("%d training jobs have completed" % job_count)
    
is_minimize = (tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['Type'] != 'Maximize')
objective_name = tuning_job_result['HyperParameterTuningJobConfig']['HyperParameterTuningJobObjective']['MetricName']
```

    20 training jobs have completed


# Describing the Best model, and its Hyperparameters


```python
from pprint import pprint
if tuning_job_result.get('BestTrainingJob',None):
    print("Best model found so far:")
    pprint(tuning_job_result['BestTrainingJob'])
else:
    print("No training jobs have reported results yet.")
```

    Best model found so far:
    {'CreationTime': datetime.datetime(2020, 12, 13, 22, 6, 26, tzinfo=tzlocal()),
     'FinalHyperParameterTuningJobObjectiveMetric': {'MetricName': 'test:binary_classification_accuracy',
                                                     'Value': 0.7124711275100708},
     'ObjectiveStatus': 'Succeeded',
     'TrainingEndTime': datetime.datetime(2020, 12, 13, 22, 11, 26, tzinfo=tzlocal()),
     'TrainingJobArn': 'arn:aws:sagemaker:us-east-1:719009365707:training-job/v3blackbelt-201213-2200-020-240888ce',
     'TrainingJobName': 'V3Blackbelt-201213-2200-020-240888ce',
     'TrainingJobStatus': 'Completed',
     'TrainingStartTime': datetime.datetime(2020, 12, 13, 22, 8, 31, tzinfo=tzlocal()),
     'TunedHyperParameters': {'bias_init_sigma': '0.00027401056839555956',
                              'bias_lr': '0.006550021000384885',
                              'bias_wd': '1.3494417573713629e-08',
                              'factors_init_sigma': '0.0006446642260676662',
                              'factors_lr': '0.0015456755116503003',
                              'factors_wd': '0.00011140124413560084',
                              'linear_init_sigma': '4.6691719892173495e-07',
                              'linear_lr': '0.002052875897255688',
                              'linear_wd': '0.00024496831939958483'}}


# Tabulating the various training Jobs and its metrics


```python
tuning_job_name='V3Blackbelt-201213-2200'
tunerresult = sagemaker.HyperparameterTuningJobAnalytics(tuning_job_name)
full_df = tunerresult.dataframe()

full_df = tunerresult.dataframe()

if len(full_df) > 0:
    df = full_df[full_df['FinalObjectiveValue'] > -float('inf')]
    if len(df) > 0:
        df = df.sort_values('FinalObjectiveValue', ascending=is_minimize)
        print("Number of training jobs with valid objective: %d" % len(df))
        print({"lowest":min(df['FinalObjectiveValue']),"highest": max(df['FinalObjectiveValue'])})
        pd.set_option('display.max_colwidth', -1)  # Don't truncate TrainingJobName        
    else:
        print("No training jobs have reported valid results yet.")
        
df

```

    Number of training jobs with valid objective: 20
    {'lowest': 0.4839620292186737, 'highest': 0.7124711275100708}





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
      <th>bias_init_sigma</th>
      <th>bias_lr</th>
      <th>bias_wd</th>
      <th>factors_init_sigma</th>
      <th>factors_lr</th>
      <th>factors_wd</th>
      <th>linear_init_sigma</th>
      <th>linear_lr</th>
      <th>linear_wd</th>
      <th>TrainingJobName</th>
      <th>TrainingJobStatus</th>
      <th>FinalObjectiveValue</th>
      <th>TrainingStartTime</th>
      <th>TrainingEndTime</th>
      <th>TrainingElapsedTimeSeconds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.740106e-04</td>
      <td>6.550021e-03</td>
      <td>1.349442e-08</td>
      <td>6.446642e-04</td>
      <td>1.545676e-03</td>
      <td>1.114012e-04</td>
      <td>4.669172e-07</td>
      <td>2.052876e-03</td>
      <td>2.449683e-04</td>
      <td>V3Blackbelt-201213-2200-020-240888ce</td>
      <td>Completed</td>
      <td>0.712471</td>
      <td>2020-12-13 22:08:31+00:00</td>
      <td>2020-12-13 22:11:26+00:00</td>
      <td>175.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.111076e-04</td>
      <td>2.368308e-02</td>
      <td>1.166224e-06</td>
      <td>1.273874e-04</td>
      <td>3.655298e-08</td>
      <td>4.483975e-02</td>
      <td>9.996352e-08</td>
      <td>8.952713e-02</td>
      <td>2.603324e-05</td>
      <td>V3Blackbelt-201213-2200-019-875a149f</td>
      <td>Completed</td>
      <td>0.710675</td>
      <td>2020-12-13 22:08:33+00:00</td>
      <td>2020-12-13 22:11:37+00:00</td>
      <td>184.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7.653696e-02</td>
      <td>8.975183e-02</td>
      <td>3.912799e+00</td>
      <td>4.140640e-07</td>
      <td>1.189309e-02</td>
      <td>7.250407e-04</td>
      <td>7.008308e-02</td>
      <td>1.530352e-01</td>
      <td>2.713528e-06</td>
      <td>V3Blackbelt-201213-2200-012-184aea36</td>
      <td>Completed</td>
      <td>0.706569</td>
      <td>2020-12-13 22:08:03+00:00</td>
      <td>2020-12-13 22:11:13+00:00</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.826342e+01</td>
      <td>8.015635e-03</td>
      <td>9.253590e-01</td>
      <td>7.466552e-08</td>
      <td>8.554517e-03</td>
      <td>1.456824e-04</td>
      <td>4.884600e-05</td>
      <td>3.360118e-04</td>
      <td>9.981838e-08</td>
      <td>V3Blackbelt-201213-2200-017-32a11ab6</td>
      <td>Completed</td>
      <td>0.705350</td>
      <td>2020-12-13 22:08:36+00:00</td>
      <td>2020-12-13 22:11:47+00:00</td>
      <td>191.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2.561240e-02</td>
      <td>3.270305e-02</td>
      <td>3.998587e-04</td>
      <td>3.917142e-06</td>
      <td>7.079697e-05</td>
      <td>6.770757e-03</td>
      <td>8.268380e-01</td>
      <td>2.254987e-01</td>
      <td>2.399299e-05</td>
      <td>V3Blackbelt-201213-2200-008-3d17c402</td>
      <td>Completed</td>
      <td>0.704196</td>
      <td>2020-12-13 22:02:51+00:00</td>
      <td>2020-12-13 22:05:43+00:00</td>
      <td>172.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4.187282e+00</td>
      <td>5.984551e-01</td>
      <td>2.646901e+00</td>
      <td>2.668094e-01</td>
      <td>1.291434e-05</td>
      <td>2.072951e+01</td>
      <td>8.070802e+00</td>
      <td>9.528089e-02</td>
      <td>3.502329e-06</td>
      <td>V3Blackbelt-201213-2200-004-cb935d25</td>
      <td>Completed</td>
      <td>0.700282</td>
      <td>2020-12-13 22:02:16+00:00</td>
      <td>2020-12-13 22:05:10+00:00</td>
      <td>174.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.887826e-01</td>
      <td>2.994439e+00</td>
      <td>3.448528e-04</td>
      <td>8.378929e-07</td>
      <td>1.244282e-07</td>
      <td>5.955022e-04</td>
      <td>1.084503e-07</td>
      <td>5.569630e-03</td>
      <td>2.763498e-06</td>
      <td>V3Blackbelt-201213-2200-016-f4f642ac</td>
      <td>Completed</td>
      <td>0.689184</td>
      <td>2020-12-13 22:08:19+00:00</td>
      <td>2020-12-13 22:11:27+00:00</td>
      <td>188.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9.020620e-08</td>
      <td>2.619983e-08</td>
      <td>1.145021e+01</td>
      <td>1.166008e-01</td>
      <td>4.270428e-05</td>
      <td>5.034846e-06</td>
      <td>1.139001e-07</td>
      <td>4.053933e-01</td>
      <td>1.434377e-03</td>
      <td>V3Blackbelt-201213-2200-006-be99fd47</td>
      <td>Completed</td>
      <td>0.647998</td>
      <td>2020-12-13 22:02:41+00:00</td>
      <td>2020-12-13 22:05:47+00:00</td>
      <td>186.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6.306584e-07</td>
      <td>1.085678e-02</td>
      <td>5.593909e-06</td>
      <td>1.414644e-08</td>
      <td>1.541493e-06</td>
      <td>5.599804e-01</td>
      <td>1.081003e-05</td>
      <td>3.136990e+01</td>
      <td>4.961945e-08</td>
      <td>V3Blackbelt-201213-2200-003-bf58c9da</td>
      <td>Completed</td>
      <td>0.644534</td>
      <td>2020-12-13 22:02:37+00:00</td>
      <td>2020-12-13 22:05:45+00:00</td>
      <td>188.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.949567e-03</td>
      <td>1.304413e-08</td>
      <td>4.066204e-04</td>
      <td>2.287576e-01</td>
      <td>2.002409e-02</td>
      <td>1.402135e-05</td>
      <td>1.429673e-02</td>
      <td>5.534390e+00</td>
      <td>3.593777e-06</td>
      <td>V3Blackbelt-201213-2200-015-6104aaba</td>
      <td>Completed</td>
      <td>0.640813</td>
      <td>2020-12-13 22:08:15+00:00</td>
      <td>2020-12-13 22:11:19+00:00</td>
      <td>184.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>5.898501e-02</td>
      <td>6.914773e-01</td>
      <td>1.604580e-01</td>
      <td>7.370527e-05</td>
      <td>1.707165e-01</td>
      <td>9.191711e-05</td>
      <td>2.539317e-04</td>
      <td>4.464457e+00</td>
      <td>2.749867e-03</td>
      <td>V3Blackbelt-201213-2200-009-4b889e87</td>
      <td>Completed</td>
      <td>0.638376</td>
      <td>2020-12-13 22:02:43+00:00</td>
      <td>2020-12-13 22:05:52+00:00</td>
      <td>189.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>6.193900e-05</td>
      <td>1.800266e-05</td>
      <td>8.216716e-08</td>
      <td>4.099914e-08</td>
      <td>1.264818e+00</td>
      <td>1.116537e-08</td>
      <td>1.223449e-03</td>
      <td>1.924276e-08</td>
      <td>4.146326e-02</td>
      <td>V3Blackbelt-201213-2200-007-279c0612</td>
      <td>Completed</td>
      <td>0.633308</td>
      <td>2020-12-13 22:02:42+00:00</td>
      <td>2020-12-13 22:05:53+00:00</td>
      <td>191.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.378268e-01</td>
      <td>1.043285e-01</td>
      <td>2.986072e+00</td>
      <td>2.513618e-08</td>
      <td>3.890103e-06</td>
      <td>7.139387e-08</td>
      <td>4.297910e-01</td>
      <td>4.320610e+00</td>
      <td>1.546842e-04</td>
      <td>V3Blackbelt-201213-2200-018-3671e9a7</td>
      <td>Completed</td>
      <td>0.614062</td>
      <td>2020-12-13 22:08:31+00:00</td>
      <td>2020-12-13 22:11:32+00:00</td>
      <td>181.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.190919e-05</td>
      <td>3.306606e-02</td>
      <td>2.580172e-04</td>
      <td>1.067043e+00</td>
      <td>1.892889e-04</td>
      <td>1.280016e-03</td>
      <td>2.100781e+00</td>
      <td>2.111876e+01</td>
      <td>3.164628e-05</td>
      <td>V3Blackbelt-201213-2200-013-9a5516e1</td>
      <td>Completed</td>
      <td>0.590454</td>
      <td>2020-12-13 22:08:50+00:00</td>
      <td>2020-12-13 22:11:47+00:00</td>
      <td>177.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5.919884e-02</td>
      <td>9.999646e-05</td>
      <td>4.148643e-07</td>
      <td>2.768268e-06</td>
      <td>3.797770e-03</td>
      <td>2.065370e-03</td>
      <td>6.827779e-03</td>
      <td>1.331972e+00</td>
      <td>1.594089e-02</td>
      <td>V3Blackbelt-201213-2200-002-6ac540f9</td>
      <td>Completed</td>
      <td>0.550488</td>
      <td>2020-12-13 22:02:21+00:00</td>
      <td>2020-12-13 22:05:23+00:00</td>
      <td>182.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.559466e+00</td>
      <td>1.326053e-02</td>
      <td>7.621253e-08</td>
      <td>6.383278e-02</td>
      <td>1.686361e+01</td>
      <td>5.252659e+00</td>
      <td>2.552739e+01</td>
      <td>4.902795e-04</td>
      <td>8.188656e-06</td>
      <td>V3Blackbelt-201213-2200-014-0aa1b0f3</td>
      <td>Completed</td>
      <td>0.519887</td>
      <td>2020-12-13 22:08:25+00:00</td>
      <td>2020-12-13 22:11:25+00:00</td>
      <td>180.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>3.345667e-08</td>
      <td>9.289300e-05</td>
      <td>1.340581e-05</td>
      <td>3.998609e-03</td>
      <td>8.416104e-07</td>
      <td>6.304549e-05</td>
      <td>1.124787e-06</td>
      <td>4.115423e-08</td>
      <td>2.033545e-08</td>
      <td>V3Blackbelt-201213-2200-001-adf7f6d6</td>
      <td>Completed</td>
      <td>0.511996</td>
      <td>2020-12-13 22:02:24+00:00</td>
      <td>2020-12-13 22:05:36+00:00</td>
      <td>192.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2.146221e-02</td>
      <td>2.572258e-02</td>
      <td>1.943825e-04</td>
      <td>1.059641e-04</td>
      <td>1.654851e-06</td>
      <td>2.474782e+01</td>
      <td>2.467029e+01</td>
      <td>3.389589e-05</td>
      <td>4.044774e-06</td>
      <td>V3Blackbelt-201213-2200-011-7f6ed714</td>
      <td>Completed</td>
      <td>0.511291</td>
      <td>2020-12-13 22:08:25+00:00</td>
      <td>2020-12-13 22:11:15+00:00</td>
      <td>170.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>9.261908e-02</td>
      <td>6.907250e-08</td>
      <td>8.978451e+00</td>
      <td>1.086359e-04</td>
      <td>5.005313e-05</td>
      <td>1.561194e-03</td>
      <td>5.598617e+00</td>
      <td>3.929962e-04</td>
      <td>7.587391e-02</td>
      <td>V3Blackbelt-201213-2200-005-8488aa47</td>
      <td>Completed</td>
      <td>0.503079</td>
      <td>2020-12-13 22:02:28+00:00</td>
      <td>2020-12-13 22:05:30+00:00</td>
      <td>182.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000855e-01</td>
      <td>2.112572e-07</td>
      <td>4.083305e-08</td>
      <td>5.853003e-07</td>
      <td>1.260105e+01</td>
      <td>3.244448e+00</td>
      <td>5.069023e-04</td>
      <td>6.484194e-01</td>
      <td>1.167850e-01</td>
      <td>V3Blackbelt-201213-2200-010-3cf125ca</td>
      <td>Completed</td>
      <td>0.483962</td>
      <td>2020-12-13 22:02:46+00:00</td>
      <td>2020-12-13 22:05:49+00:00</td>
      <td>183.0</td>
    </tr>
  </tbody>
</table>
</div>



# Plotting the graph with results from Hyperparameter tuner


```python
import bokeh
import bokeh.io
bokeh.io.output_notebook()
from bokeh.plotting import figure, show
from bokeh.models import HoverTool

class HoverHelper():

    def __init__(self, tuning_analytics):
        self.tuner = tuning_analytics

    def hovertool(self):
        tooltips = [
            ("FinalObjectiveValue", "@FinalObjectiveValue"),
            ("TrainingJobName", "@TrainingJobName"),
        ]
        for k in self.tuner.tuning_ranges.keys():
            tooltips.append( (k, "@{%s}" % k) )

        ht = HoverTool(tooltips=tooltips)
        return ht

    def tools(self, standard_tools='pan,crosshair,wheel_zoom,zoom_in,zoom_out,undo,reset'):
        return [self.hovertool(), standard_tools]

hover = HoverHelper(tunerresult)

p = figure(plot_width=900, plot_height=400, tools=hover.tools(), x_axis_type='datetime')
p.circle(source=df, x='TrainingStartTime', y='FinalObjectiveValue')
show(p)

```



<div class="bk-root">
    <a href="https://bokeh.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
    <span id="2042">Loading BokehJS ...</span>
</div>











<div class="bk-root" id="d4948d52-10b1-4e6e-a564-fb4ace881edc" data-root-id="2044"></div>





# Plotting the graph with results from Hyperparameter tuner


```python
ranges = tunerresult.tuning_ranges
figures = []
for hp_name, hp_range in ranges.items():
    categorical_args = {}
    if hp_range.get('Values'):
        # This is marked as categorical.  Check if all options are actually numbers.
        def is_num(x):
            try:
                float(x)
                return 1
            except:
                return 0           
        vals = hp_range['Values']
        if sum([is_num(x) for x in vals]) == len(vals):
            # Bokeh has issues plotting a "categorical" range that's actually numeric, so plot as numeric
            print("Hyperparameter %s is tuned as categorical, but all values are numeric" % hp_name)
        else:
            # Set up extra options for plotting categoricals.  A bit tricky when they're actually numbers.
            categorical_args['x_range'] = vals

    # Now plot it
    p = figure(plot_width=500, plot_height=500, 
               title="Objective vs %s" % hp_name,
               tools=hover.tools(),
               x_axis_label=hp_name, y_axis_label=objective_name,
               **categorical_args)
    p.circle(source=df, x=hp_name, y='FinalObjectiveValue')
    figures.append(p)
show(bokeh.layouts.Column(*figures))
```








<div class="bk-root" id="6f151ea0-d854-4ee3-a290-d557ae47093d" data-root-id="2704"></div>






```python

```
