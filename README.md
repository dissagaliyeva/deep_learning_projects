# Udacity Deep Learning Nanodegree Program + Own Projects
<img src="readme_images/cert.jpg" width="500px" height="500px"/>
This folder contains all the projects from Udacity's Nanodegree Program along with own projects. The major projects are 
stored in the 'project-XXX' format or below:

**Udacity**
- [x] [Bike-Sharing (pure numpy)](#bike-sharing)
- [x] [Landmark Classification (CNN, Transfer Learning, VGG16)](#landmark)
- [x] [TV Script Generator (RNN, LSTM)](#tv-script-generator) 
- [x] [Face Generator (GAN, CycleGAN)](#face-generator) 
- [x] [Deploying Sentiment Analysis Model (AWS SageMaker, REST API, NLP)](#deploying-the-model)

**Own**
- [x] [102 Flowers (CNN, Transfer Learning, ResNet34, Discriminative Learning Rates)](#flower)
- [] Bike-Sharing (CNN, Regression) 
- [] Dog Classification (CNN, Transfer Learning) 
- [] Cancer Detection (CNN, Transfer Learning) 


## Bike-Sharing Rentals <a class="anchor" id="bike-sharing"/> <br>
* <a href="https://github.com/issagaliyeva/deep_learning_projects/blob/master/project-bike-sharing/Predicting_bike_sharing_data.ipynb">Numpy Implementation</a>
* <a href="https://github.com/issagaliyeva/deep_learning_projects/blob/master/project-bike-sharing/pytorch_implementation.ipynb">Pytorch Implementation</a> [under development]

<b>The first</b> case was implemented using a vanilla approach with pure numpy package. <br>
<b>The second one</b>, however, is my implementation to see how the results differ using PyTorch.  

<b>Project Description</b>
* hour.csv: bike sharing counts aggregated on hourly basis. Records: 17379 hours
* day.csv: bike sharing counts aggregated on daily basis. Records: 731 days

Bike sharing systems are new generation of traditional bike rentals where whole process from membership, rental and return 
has become automatic. Through these systems, user is able to easily rent a bike from a particular position and return 
back at another position. Currently, there are about over 500 bike-sharing programs around the world which is composed of 
over 500 thousands bicycles. Today, there exists great interest in these systems due to their important role in traffic, 
environmental and health issues.

### Results
![Bike Sharing Final Predictions](readme_images/bike-sharing.png) <br>
The objective of the project was to predict the bike rentals within the last month (December) of the second year.
It managed to achieve 0.062 on training set and 0.133 on test set. However, we can see that the December's results are overshooting.
The majority of people use bikes for work commutes. As we can see, the data we're testing the results on the last days of December. As we saw above, there is an over-prediction in the last days of December. 
We see such a phenomenon due to the fact that the Neural Network has seen one-year record (since we're withholding the last year for testing purposes). It could not generalize because of the holiday season. The solution to which would be to include more data on consecutive years.

---

## Landmark Classifier <a class="anchor" id="landmark"/>
* <a href="https://github.com/issagaliyeva/deep_learning_projects/blob/master/project-landmark-classifier/landmark.ipynb">VGG16 Model</a>
* <a>Other Models</a> [under development]

This project is subset of <a href='https://www.kaggle.com/google/google-landmarks-dataset'>Google Landmarks</a> dataset on Kaggle. The objective of this project was to build own and transfer architectures using Convolutional Neural Networks (CNN).

<b>Project Description</b>
* 5000 images (50 folders, 100 images in each)
* Datasets can be found here (BE CAREFUL, automatic download): [landmark dataset](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip)

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgement to classify these landmarks would not be feasible.

### Results

<img src="readme_images/landmark.png" align="left"/>
After going through dozens of research papers on Convolutional Neural Networks, I found the optimal architecture that managed to 
score 41% instead of the passing score of 20%. After reading <a href="https://arxiv.org/abs/1409.4842">Going Deeper with Convolutions</a>, I've decided to
go with 5-layer convolutions (each followed by ReLU activation function and max pooling) and 3 fully-connected layers. The interesting
part is that having 4 convolutional layers had 24% on the test data. By adding one more layer, its accuracy jumped by 17%! 

However, using VGG-16 for pre-trained model, scored a bit more than 70%. Not only did it score moderately, but also has over 20 million parameters. 
This finding led me to reconsider my choice and find more optimal models. The experimentation will be covered in the next notebook.

---

## TV Script Generator
* <a href="https://github.com/issagaliyeva/deep_learning_projects/blob/master/project-tv-script-generation/dlnd_tv_script_generation.ipynb">RNN & LSTM implementation</a><br>

<b>Project Description</b>
* Seinfeld_Scripts.txt: All 9 season scripts (3.41 MB)

Imagine you are working for a production company, and your job is to write a script for one of their shows. You could write it manually or let an algorithm do it for you! In this
project we are asked to use RNN along with LSTM generate a good model whose accuracy should be <= 3.5.  <br>

### Results
The model I've created barely passed the 3.5 threshold and had the following parameters:
- epochs: 10, sequence length: 8, batch size: 256, learning rate: 0.001
- hidden dimensions: 400, embedding dimensions: 300

The interesting fact that I've noticed was that having smaller learning rate, sequence length and batch size gave much better 
performance. Even though the lectures mentioned that the optimal batch size is in the range of 32 to 512, for this particular 
assignment using 64-100 batch size yielded in the best performance. I'll need to spend more time with this topic and return to it
with better understanding :)

**Small results extract**
```
jerry: and the best thing i ever heard.

jerry:(to the phone) hello, hello.

jerry: hey, you don't know?

jerry: yeah!

george: what happened to you?

jerry: i was just trying to get the money.

jerry: oh, come on!(he takes the magazine on the table and starts walking into her room) : yeah.

george: i mean, i was just trying to get some sleep.

jerry: oh, come on. i don't think so.(to jerry) you know, i don't want to talk about it...

kramer: oh, yeah... i got the car.

george: i don't want to be here.
```


---
## Face Generator
* <a href="">CycleGAN</a>


---
## Deploying the Model
* <a href="https://github.com/issagaliyeva/deep_learning_projects/blob/master/project-aws-deployment/SageMaker%20Project.ipynb">SageMaker + REST API (for app)</a><br>
This project was an extension of lectures where


---

## 102 Flower Classifier <br>
<a href="https://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.pdf">Official Paper</a>&emsp;<a href="https://www.robots.ox.ac.uk/~vgg/data/flowers/102/">Official Page</a>&emsp;<a href="https://s3.amazonaws.com/fast-ai-imageclas/oxford-102-flowers.tgz">Download Dataset</a>&emsp;<a href="https://gist.github.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1">Download Labels</a>
<br>

<b>Project Description</b>
* train.csv : 1020 flower images' paths and targets (10 instances per category)
* validation.csv: 1020 flower images' paths and targets (10 instances per category)  
* test.csv: 6149 flower images' paths and targets

Although the idea of Deep Learning was born almost 80 years ago, we are just given the opportunity to bring any 
sophisticated ideas to life. That is why we will be looking 
at the extensively researched case of classfying 
<a href="https://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.pdf">102 categories of flowers</a>. 
Even though the task does not sound hard, the challenges arise in the similarity in both color and shape.

<img src="readme_images/models.png" width="350px" align="left"/> <br>
To give an example of similarities, there are nine types of lilies (Giant White Arum, Fire, Water, Toad, Blackberry, 
Sword, Tiger, Peruvian, Canna) and two types of 
irises (Bearded and Yellow). Additional difficulties can be found with the non-conventional distribution of training 
(12.45%), validation (12.45%), and testing (75.1%) sets.
The project will be worked with <a href="https://arxiv.org/abs/1512.03385">ResNet34</a> pre-trained model (a residual 
network that was trained on ImageNet's images).
It is considerately smaller than most of the other models with only 20 million parameters. Thus, the training part 
will not take long even on CPU mode (from my findings, running 15 epochs takes around 40 minutes). 


### Results
The notebook contains experimentations with different model instantiations, in particular, various optimizers. 
There are three of them used: <a href="https://paperswithcode.com/method/sgd">Stochastic Gradient Descent</a> (<a href="https://arxiv.org/abs/1607.01981">Nesterov's Accelerated Gradients</a>), <a href="https://paperswithcode.com/method/adagrad">Adagrad</a> and <a href="https://paperswithcode.com/method/adam">Adam </a>. It's challenging to find the correct optimizers,  so I wanted to see the actual differences in prediction before settling with a specific one. In addition to these experimentations, fellowship.ai's requirement was to use Discriminative Learning Rates. Thus, all models were trained using that technique along with Exponential Learning Decay Scheduler to boost accuracy. 

The overall accuracy of each model is over 80%, with the best ones (SGD and Adam) scoring 87%. However, there is the fifth model that has more transformations, which had a 100%
confidence on classifying the never-seen-before image of a sunflower. 

### What I learnt
* os package <br>
I was manually transferring the images to satisfy DataLoader's and ImageFolder's requirements. Therefore, there was an extensive usage of such package. 
The main feature to remember: **DataLoader treats names as strings, even if the names are numeric**. PyCharm uses automatic folder sorting, so there might be
mismatch between PyCharm's and DataLoader's way of structuring things.   <img src="readme_images/differential.png" width="300px" align="right"/>
* Discriminative Learning Rates  <br>
To better understand how Discriminative Learning Rates work, the following image might help (right). 
To implement this technique, we need to use smaller learning rates at convolutional layers and bigger ones in the fully-connected ones. Doing so boosted the performance. 
I specifically want to focus on the case of Adagrad. At the beginning, I've trained it using 
<a href="https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR">Exponential Learning Decay Scheduler</a>. 
By doing so, only **depressing** 23% of the images were correctly classified. However, by changing per-layer learning rates resulted in 61% performance improvement! 
 

---