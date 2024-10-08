# HMILungTestCode

This repo contain test codes for lung tumor detection task under Hyperspectral Microscopic Imaging (HMI). The paper is under review.

Please download data, label and model params in Baidu Cloud Disk as follow, and put the content in the corresponding folder in this repo:

Link: https://pan.baidu.com/s/1noGf1BR9l4953ib8AZQy7A?pwd=fqq3 
Password: fqq3 

# File Structure
configs: contain configurations used in test, such as directory position. The default option is to test Hybrid_BN_Attention model only. If you want to test other model, set the corresponding "test" field to 1.
data: contain input hyperspectral datacube
label: contain labels for 4 types. Background, non-cell, tumor cell and non-tumor cells.
models: definition for models.
params: model parameters.
ResultProb: Predicted probability matrix for each pixel and each type
ResultLabel: Predicted label under different post-processing strategy
test.py and utils.py: main codes used for testing
postprocess.py: generate direct, noisy and final post-processed prediction.
postprocess_direct.py: generate direct prediction only (highest probability in ResultProb)

# How to run
1. Download data, label and params from cloud disk and put them into the folders with same name in this repo.

2. run test.py to generate predicted probability in ResultProb folder

3. run postprocess.py or postprocess_direct.py to generate predicted label.

You can change configurations when needed.

# Contact
If any problem occurs, please leave messages in Issues and we will respond ASAP.

Corresponding Author

Yunfeng Nie Yunfeng.Nie@vub.be

Jingang Zhang zhangjg@ucas.ac.cn

Authors

Zhiliang Yan yz97liang@stu.xidian.edu.cn

Haosong Huang hhuang2@stu.xidian.edu.cn (Primary Contact)
