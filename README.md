# TA_feature_FT
H. Zeng, B. Chen, A. Peng, Enhancing targeted transferability via feature space fine-tuning, [arXiv](https://arxiv.org/abs/2401.02727) 2024ICASSP, Seoul. We sincerely invite interested researchers to discuss relevant issues in person.

### Note
Our method works in an add-on manner. The attack ability of the final AE is mainly determined by the baseline attack you set, not the proposed feature-space fine-tuning (FFT). Hence, it _does not_ make sense to directly compare our method with SOTA baseline attacks, such that: FFT vs. Logit, or FFT vs. SU.　  
A good practice to test FFT is: CE+FFT vs. CE, Logit+FFT vs. Logit, or SU+FFT vs. SU.

Following are the results when Resnet50 as the surrogate:
| method\victim     | Inc-v3 | Dense121 | VGG16 |
| -------------     | ------ |--------- |------ |
| CE (200 iters)    | 3.9    | 44.9     | 30.5  |
| CE+FFT (160+10)   | 9.0    | 60.4     | 49.3  |
| Logit (200 iters) | 9.1    | 70.0     | 61.9  |
| Logit+FFT (160+10)| 15.8   | 75.3     | 64.1  |
 
Even after fine-tuning, CE+FFT lags behind Logit, which is expected.

### Dataset
The 1000 images are from the NIPS 2017 ImageNet-Compatible dataset. [official repository](https://github.com/cleverhans-lab/cleverhans/tree/master/cleverhans_v3.1.0/examples/nips17_adversarial_competition/dataset) or [Zhao's Github](https://github.com/ZhengyuZhao/Targeted-Tansfer/tree/main/dataset). 

### Evaluation
We use the proposed fine-tune strategy to improve five state-of-the art simple transferable attacks:   
CE,   
Po+Trip,  
[Logit](https://github.com/ZhengyuZhao/Targeted-Transfer),   
[SupHigh](https://github.com/zengh5/Transferable_targeted_attack).  
[SU](https://github.com/zhipeng-wei/Self-Universality).  
All attacks are integrated with TI, MI, and DI. Baseline attacks are run with _N_=200 iterations to ensure convergence. When fine-tuning is enabled, we set _N_=160, _N<sub>ft</sub>_=10, to make the running time with/without fine-tuning comparable.
L<sub>&infin;</sub>=16 is applied.

#### ```main_feature_FT.py```: feature space fine-tuning in the random-target/most-difficult target scenarios.
#### ```main_feature_FT_tUAP```: Fine-tune date-free targeted UAPs. 

### More results
We provide ablation study on fine-tuning iterations _N<sub>ft</sub>_ and target layer _k_, visual comparison, and the results of date-free targeted UAP, in the 'supp.pdf'. We also provide the results when an alternative method (RPA) is used for calculating the aggregate gradient.

### Acknowledgement
a) The f(I'<sub>ft</sub>) in Eq. (7) should be f<sub>k</sub>(I'<sub>ft</sub>). Thanks to A. Peng.  
b) We mis-clarify the learning rate (self.alpha) used in the fine-tuning phase. In our implementation, we adaptively set it as self.alpha=self.epsilon / 16, which is slightly different to the fixed learning rate (2 for epsilon=16) used in baseline attacks. Thanks to atnegam.
