---
title: 'Introduction to Comet.ml'
date: 2020-04-04
permalink: /posts/python-intro-to-comet/
description: A brief introduction to the Python SDK of Comet.ml, and how to parse results from the dashboard
tags:
  - Python
  - PyTorch
  - Comet.ml
  - Introduction
  - REST API
  - Machine learning
---

[Comet](https://www.comet.ml/) is a neat alternative to the classical Tensorboard experiment tracker, which supports multiple functions such as logging of experiment metrics, plots, gradients, model weights, as well as an online dashboard similar to that of Tensorboard. There are other similar products such as Weights&Biases, but Comet has the best support on the Compute Canada clusters.

The module can be installed from PyPi using
```
pip install comet_ml
```
and then used very, very trivially:
```
from comet_ml import Experiment

dummy_parameters = {'learning_rate':1e-3,
                    'dropout':True}
API_KEY = 'your api key'
experiment = Experiment(api_key=API_KEY,project_name='dummy-run',workspace="your username")
experiment.set_name( 'This will be the name of your experiment' )
experiment.log_parameters( dummy_parameters )
```

To work with Comet (even offline), you require an API key. It can be found on your dashboard, in the top right corner. You should then specify a project name, as well as your workspace which, for most plans, is simply your username.

Depending on the plan that you're using, you might get access to more or less features. The *Academic* plan is extremely convenient, providing free access to functionalities not available to the free tier otherwise. Find it at [https://www.comet.ml/site/academics/](https://www.comet.ml/site/academics/).

## Logging your runs
You can now start running your simulations, with the only addition being that, to log a metric such as loss or accuracy, you would write
```
experiment.log_metric("loss_train",0.02,step=1000)
```
Alternatively, you can instead log the epoch.

You can log models by first saving them into, say, a `.pickle` file, and subsequently calling
```
experiment.log_model("resnet-finetuned", "models/train/resnet-finetuned-100")
```
This will be stored in the dashboard under every experiment, `Assets->others`.

Similarly, images and weight histograms can be saved using the `experiment.log_*` functions.

## Retrieving your runs
So far, Comet provides a REST api with a Python interface which allows you to download everything found on the dashboard, including metrics, models and figures.

This code snippet populates `exps` with all experiments from a given project
```
from comet_ml import API

API_KEY = 'your api key'

comet_api = API(api_key=API_KEY)

exps = comet_api.get("your username",'dummy-run')
```

This returns a list of experiments (i.e. curves on the dashboard), which you can then iterate over:
```
import matplotlib.pyplot as plt
for i,exp in enumerate(exps):
    run_params = {}
    
    for param in exp.get_parameters_summary():
        run_params[ param['name'] ] = param['valueCurrent]
    
    x,y = [], []
    for pt in exp.get_metrics('loss_train'):
        y.append( float(pt['metricValue']) )
        x.append( float(pt['step']) )
    plt.plot(x,y)
```

Comet assumes by default that all parameters can change across experiments, which is why every parameter's value is stored in the `valueCurrent` key. Iterating over paramters and re-writing them into a separate `dict` does the job.

Then, for every curve (i.e. experiment) on the dashboard, you can access any metric by its name. For example, `exp.get_metrics('loss_train')` returns a list of dictionaries, where every item corresponds to a data point. You can the interate over them to extract proper x and y values.

Retrieving model weights from the Comet's cloud storage is a little trickier, since Python seems to have issues with the command they provide.

Instead, here is a snippet to use the `wget` module from Python directly, without any call to the `os.system`:
```
import os

assets = exp.get_asset_list() 
model_path = 'model'
for asset in assets:
    cmd = asset['curlDownload'].split('>') 
    cmd[1] = os.path.join(model_path,cmd[1].lstrip().rstrip()) 
    wget.download(cmd[0].split(' ')[1],cmd[1])
```
This command queries the URL provided by the REST API through `wget`, and saves the file under `model_path/`.

------