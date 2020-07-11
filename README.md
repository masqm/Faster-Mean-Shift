# Faster_Mean_Shift

Faster Mean-shift algorithm for accelerating the recurrent neural network (RNN) based pixel embedding framework for holistic cell segmentation and tracking. Here is a brief introduction on how to run it.


##  Environment
Win10

VS2019

Anacoda 2020.02

The packages requirement please see requirements.txt


## Preparation
1. Download the datasets from the [celltracking challenge](http://www.celltrackingchallenge.net/) and extract them under a ***input_path***. 
2. Download the corresponding trained model from my [Google Driver](???) and put them under a ***model_path***.
3. Creat a folder for ***output_path***
4. Specify the ***dataset_name*** you want to run. Celltracking challenge provides 7 data-sets. We have tested four of them：

（1）DIC-C2DH-HeLa

（2）Fluo-N2DH-GOWT1

（3）Fluo-N2DH-SIM+

（4）PhC-C2DH-U373

## Modify Variable
In order to run the program, you need to modify some variables in segment_and_track.py

    input_image_folder = input_path		#eg."E:/code/data/myfile/01"
    
    output_folder = output_path 		#eg."E:/code/data/myfile/out"
    
    model_file_name = model_path		#eg."E:/code/data/myfile/01_model/DIC-C2DH-HeLa"
    
    dataset_name = dataset_name 		#eg."DIC-C2DH-HeLa"

## Testing
We provided a vs2019 project file for testing. You can runing the program by excuate modified segment_and_track.py. We evaluated the time consumption and GPU memory requirement of the program. Please read our paper for specific performance data. 

## Migration Algorithm
You are very welcome to use our faster mean-shift algorithm to develop your program. The entire algorithm is based on the following two files:

[mean_shift_cosine_gpu.py](https://github.com/masqm/Faster_Mean_Shift/blob/master/utils/mean_shift_cosine_gpu.py "mean_shift_cosine_gpu.py")

[batch_seed.py](https://github.com/masqm/Faster_Mean_Shift/blob/master/utils/batch_seed.py "batch_seed.py")

An example of using the algorithm is given below：

    #Import our algorithm
    from  utils.mean_shift_cosine_gpu  import  MeanShiftCosine
    
    #Configuration
    cluster = MeanShiftCosine(bandwidth=0.1, cluster_all=True, GPU=True)
    
    #Clustering the vectors. x is input vector. The algorithm will it.
    cluster.fit(x)
    
    #Obtain result
    labels  =  cluster.labels_
