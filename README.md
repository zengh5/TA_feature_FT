# TA_feature_FT
Enhancing targeted transferability via feature space fine-tuning

### Dataset
The 1000 images are from the NIPS 2017 ImageNet-Compatible dataset. [official repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset) or [Zhao's Github](https://github.com/ZhengyuZhao/Targeted-Tansfer/tree/main/dataset). 

### Evaluation
We use the proposed fine-tune strategy to improve four state-of-the art simple transferable attacks:   
CE,   
Po+Trip,  
[Logit](https://github.com/ZhengyuZhao/Targeted-Transfer),   
[SupHigh](https://github.com/zengh5/Transferable_targeted_attack).  
All attacks are integrated with TI, MI, and DI. Baseline attacks are run with N=200 iterations to ensure convergence. When fine-tuning is enabled, we set N=160, N_ft=10, to make the running time with/without fine-tuning comparable.
L<sub>&infin;</sub>=16 is applied.

#### ```main_feature_FT.py```: feature space fine-tuning in the random-target/most-difficult target scenarios.
#### ```main_feature_FT_tUAP```: Fine-tune date-free targeted UAPs. 

### More results
We provide ablation study on fine-tuning iterations N_ft and target layer k, visual comparison, and the results of date-free targeted UAP, in the 'supp.pdf'.

