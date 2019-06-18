# EvasionAttacksSDC

The paper demonstrates for the first time evasion attacks against both classification and regression models used in autonomous vechicle domain.

Language: python 3.6<br/>	
Dependencies: numpy, tensorflow, keras, imageio, skimage


## Acknowledgements

Firstly, I need to acknowledge chrisgundling and rwightman.

*	I've used chrisgundling instructions for data extraction( https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/cg23)
	and his Epoch nodel architecture for training the classifier.
*	I've used rwightman’s reader (https://github.com/rwightman/udacity-driving-reader) for extracting the ROS .bag files.
  
## Project structure
*	classification/
	*	results/
		*	txt files with attack results
		*	png files with result plots
	*	models/
		*	sdc_epoch
		*	sdc_nvidia
	*	attack.py
	*	model.py
	*	utilities.py
	*	train_epoch_model.py
	*	train_nvidia_model.py
	*	attack_epoch.py
	*	attack_nvidia.py
	*	success_epoch.py
	* success_nvidia.py
	*	roc_epoch_py
	* roc_nvidia.py
	*	adv_images_epoch.py
	*	adv_images_nvidia.py
*	regression/
	* results/
		*	txt files with attack results
		*	png files with results plots
	*	models/
		*	sdc
	*	attack.py
	*	model.py
	*	utilities.py
	*	epoch_attack.py
	*	regression_epoch_training.py
	* results_plots.py
		

## Data

The data collected from the Udacity car was images (mostly .jpg) from 3 cameras (left, center, right spaced 20 inches apart) and steering angles.
There is a number of datasets, but we've used the one, which is a combination of highway driving and curvy driving over Highway 92 from San Mateo to Half Moon Bay.
*	Dataset: https://github.com/udacity/self-driving-car/tree/master/datasets/CH2 

For data extraction I followed the chrisgundling guide, which consists of following steps:
* Installing Docker on virtual box
*	Creating a shared directory from the virtual box to the local machine where the .bag files are stored,
*	Running the rwightman’s run-bagdump.sh script which generated directories with the data.

These are the steps:
* Download VirtualBox
*	Install Ubuntu 16.04 on VirtualBox
*	Install Docker with this link: https://docs.docker.com/engine/installation/linux/ubuntulinux/
*	Download rwightman’s repository to VirtualBox
*	Download transmission for torrents: http://transmissionbt.com
*	Download the torrents to your local machine
*	Setup shared directories between your VirtualBox and local machine/external storage
*	Run rwightman’s reader

For training we used only images from center camera, the corresponding images can be found in center/ folder.
All information about images, including corresponding steering angles is in interpolated.csv file.

## Training
Training Epoch Classification task:<br/>
*	Specify the path to the interpolated.csv file(IMAGE_FILE = 'path to interpolated.csv') and center/ folder(IMAGE_FOLDER = 'path to the center/ folder')
in train_epoch_model.py file under classification/ folder
*	run train_epoch_model.py

The resulting trained model 'sdc_epoch' will be in models/ folder.<br/>

Training Nvidia Classification task:<br/>
*	Specify the path to the interpolated.csv file(IMAGE_FILE = 'path to interpolated.csv') and center/ folder(IMAGE_FOLDER = 'path to the center/ folder')
in train_nvidia_model.py file under classification/ folder
*	run train_nvidia_model.py

The resulting trained model 'sdc_nvidia' will be in models/ folder.

Training Epoch Regression task:<br/>
*	Specify the path to the interpolated.csv file(IMAGE_FILE = 'path to interpolated.csv') and center/ folder(IMAGE_FOLDER = 'path to the center/ folder')
in regression_epoch_training.py file under regression/ folder
*	run regression_epoch_training.py

The resulting trained model 'sdc' will be in models/ folder.

## Performing Attack

In order to run the attack against 'sdc_epoch' classification model:
*	Create the .csv file with the same column names as in interpolated.csv files and paste the information about images, against 
which you want to run the attack, create the folder with corresponding images
*	Specify the path to this .csv file(IMAGE_FILE = 'path to .csv file')  and folder with images(IMAGE_FOLDER = 'path to the folder')
in attack_epoch.py file under classification/ folder
* 	Specify the number of images, which are used to run the attack (NUM_ATTACKS field)
*	run attack_epoch.py

This will create the following files with results under results/ folder:
*	'res_attack_epoch.txt' - file with resulting distances from input file
*	'res_attack_success_epoch.txt' - file with the number of successfull attacks
*	'res_attack_probas_epoch.txt'- file with the resulting adversarial class probabilites
* 'res_attack_probas_labels_epoch.txt' - file with predicted legitimate class probabilities
epoch_success_rate.png


In order to run the attack against 'sdc_nvidia' classification model:
*	Create the .csv file with the same column names as in interpolated.csv files and paste the information about images, against 
which you want to run the attack, create the folder with corresponding images
*	Specify the path to this .csv file(IMAGE_FILE = 'path to .csv file')  and folder with images(IMAGE_FOLDER = 'path to the folder')
in attack_nvidia.py file under classification/ folder
* 	Specify the number of images, which are used to run the attack (NUM_ATTACKS field)
*	run attack_nvidia.py

This will create the following files with results under results/ folder:
*	'res_attack_nvidia.txt' - file with resulting distances from input file
*	'res_attack_success_nvidia.txt' - file with the number of successfull attacks
*	'res_attack_probas_nvidia.txt'- file with the resulting adversarial class probabilites
* 	'res_attack_probas_labels_nvidia.txt' - file with predicted legitimate class probabilities


In order to run the attack against 'sdc' regression model under results/ folder:
*	Create the .csv file with the same column names as in interpolated.csv files and paste the information about images, against 
which you want to run the attack, create the folder with corresponding images
*	Specify the path to this .csv file(IMAGE_FILE = 'path to .csv file')  and folder with images(IMAGE_FOLDER = 'path to the folder')
in attack_epoch.py file under regression/ folder
*	run epoch_attack.py

This will create the following files with results under results/ folder:
*	'res_attack_mse_ratio.txt' - file with the list of ratios of legitimate MSE over adversarial MSE
*	'res_attack_mse_no_attack.txt' - file with the MSE values for the legitimate images
*	'res_attack_mse_with_attack.txt'- file with the MSE values for the adversarial images
* 	'res_attack_distance.txt' - file with the resulting L2 norm distances

For the classification case there is 'straight_right_left.csv' file with the information about imaged from all three classes,
that can be used to run the attack against classification model.

## Results Plots
In order to get the success rate plot wrt the amount of perturbation added by the adversary(classification task):
*	run success_epoch.py or success_nvidia.py under classification/ folder
*	it is possible to additionally change the distance thresholds in these files corresponding to 'more_*' fields

This will produce epoch_success_rate.png/nvidia_success_rate.png files under results/ folder.


In order to get micro-average ROC curves wrt different L2 norm distances(classification case):
* run roc_epoch.py or roc_nvidia.py files under classification/ folder 

This will produce epoch_roc.png/nvidia_roc.png files under results/ folder.

In order to get results plots for the attack against 'sdc' regression model:
*	run results_plots.py under regression/ folder

This will produce cdf_mse.png file under results/folder with CDF for the case with and without the presence of attack.



## Adversarial Examples

In order to produce the adversarial example against sdc_epoch'/'sdc_nvidia' classification model:
*	Create the .csv file with the same column names as in interpolated.csv files and paste the information about images, against 
which you want to run the attack, create the folder with corresponding images
*	Specify the path to this .csv file(IMAGE_FILE = 'path to .csv file')  and folder with images(IMAGE_FOLDER = 'path to the folder')
in adv_images_epoch.py/adv_images_nvidia.py file under classification/ folder
*	run adv_images_epoch.py/adv_images_nvidia.py

Under classification folder there is 'epoch.csv' and 'nvidia.csv' files with the information about images that can be modfied inorder to create adversarial examples against epoch and nvidia model respectively.
In this case IMAGE_FILE = 'nvidia.csv' or 'epoch.csv' IMAGE_FOLDER = 'path to center/ folder'

This will produce images for the input files and corresponding adversarial examples under results/ folder.


In order to produce the adversarial example against 'sdc' regression model:
*	Create the .csv file with the same column names as in interpolated.csv files and paste the information about images, against 
which you want to run the attack, create the folder with corresponding images
*	Specify the path to this .csv file(IMAGE_FILE = 'path to .csv file')  and folder with images(IMAGE_FOLDER = 'path to the folder')
in adv_images.py file under regression folder.
*	run adv_images.py

Under regression folder there is 'test_image.csv' file for adversarial example against epoch regression model.
In this case IMAGE_FILE = 'test_image.csv' and IMAGE_FOLDER = 'path to center/ folder'.


This will produce images for the input files and corresponding adversarial examples under results/ folder.





