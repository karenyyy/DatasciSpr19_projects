## Homework 2: Image Segmentation Algorithms From Scratch


- OS: Ubuntu 18.04

- Python Version: 3.6.7

- PyQt Version: 5


### Algorithms Written from scratch

- Kmeans Clustering (k could be adjusted by users in the GUI)

- Meanshift Clustering (radius/bandwidth could be adjusted by users in the GUI)
    - only implemented with feature (color) bandwidth

- Region Growing

- Gaussian Mixture Model
    - (*) __break down the math__:
        - E step:
            for each point compute:
            $$q(t_i) = p(t_i \mid x_i, \mu, \sigma)$$
        - M step: update Gaussian parameters to fit points assigned to them
        	- how to get the updated $\mu$ and $\sigma$?

objective function:

Denote:

$t$: the latent variable introduced, it refers to the label of each gaussian distribution in GMM
$$\max_{\theta} \sum^N_{i=1} \mathbb{E}_{q(t_i)} \log p(x_i, t_i \mid \mu,  \sigma) = $$

$$\sum^N_{i=1} \sum^C_{c=1} q(t_i = c) \cdot \log (\exp (- \frac{x_1 - \mu_c}{2 \sigma^2}) \cdot \pi_c)=$$

$$\sum^N_{i=1} \sum^C_{c=1} q(t_i = c) \cdot (\log \pi_c - \frac{(x_i - \mu_c)^2}{2 \cdot \sigma_c^2})$$

then since the lower bound is concave function, thus arg max could be obtained by setting gradient to 0:

$$\frac{\partial}{\partial \mu_i} = \sum^N_{i=1} q(t_i = c) \cdot \frac{(x_i - \mu_c)}{\sigma_c^2} = 0$$

$$\sum^N_{i=1} q(t_i = c) \cdot (x_i - \mu_c) = 0$$

thus:

$$\mu_c = \frac{\sum_i p(t_i=c | x_i, \mu, \sigma) x_i}{\sum_i p(t_i = c| x_i, \mu, \sigma)}$$
        
similarly for $\sigma$:

$$\sigma_i^2 = \frac{\sum^N_{i=1} (x_i - \mu_c)^2 \cdot q(t_i = c) }{\sum^N_{i=1} q(t_i = c)}$$

subj to:
$$\pi_1 + \pi_2 + \pi_3 = 1$$

where:
$$\pi_c = \frac{\sum^N_{i=1} q(t_i = c)}{N}$$

### Algorithms Written using functions from OpenCV

- GrabCut

- Watershed

### Test 

```python
python3 main.py
```
* see hw2/test.mp4