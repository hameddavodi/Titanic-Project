## Definition of bootstrap:

The Bootstrap method is a statistical resampling technique that involves drawing multiple samples with replacement from an original data set. Each of these samples is used to estimate some statistic of interest, such as the mean or standard deviation. By repeating this process many times, we can obtain a distribution of estimates for the statistic, which can be used to calculate confidence intervals and other measures of uncertainty.

Here's an example of how to implement the Bootstrap method in Python to estimate the mean of a sample:

```python
import numpy as np

# Generate sample data
np.random.seed(123)
sample = np.random.normal(loc=10, scale=2, size=100)

# Calculate the mean of the sample
sample_mean = np.mean(sample)

# Generate 1000 bootstrap samples
n_bootstraps = 1000
bootstrap_means = []
for i in range(n_bootstraps):
    # Draw a bootstrap sample with replacement
    bootstrap_sample = np.random.choice(sample, size=len(sample), replace=True)
    # Calculate the mean of the bootstrap sample
    bootstrap_mean = np.mean(bootstrap_sample)
    # Store the result
    bootstrap_means.append(bootstrap_mean)

# Calculate the 95% confidence interval for the mean
confidence_interval = np.percentile(bootstrap_means, [2.5, 97.5])
```
In this example, we first generate a sample of 100 values drawn from a normal distribution with mean 10 and standard deviation 2. We then calculate the mean of the sample using the `np.mean()` function.

Next, we generate 1000 bootstrap samples by drawing 100 values with replacement from the original sample using the `np.random.choice()` function. For each bootstrap sample, we calculate the mean using `np.mean()` and store the result in a list called bootstrap_means.

Finally, we calculate the 95% confidence interval for the mean using the `np.percentile()` function. The confidence interval is calculated as the 2.5th and 97.5th percentiles of the bootstrap means.

```python
confidence_interval
```
array([ 9.57324432, 10.49708713])

and we can plot the bootstrap means with:
```python 
sns.displot(bootstrap_means)
```

![Untitled](https://user-images.githubusercontent.com/109058050/230631863-c40e35be-b570-4b24-8b77-cdf6e348de1f.png)

Note: The bootstrap is a method for estimating the sampling distribution of a statistic by resampling data from an observed sample, with replacement. It is not a type of central limit theorem, but rather a computational technique that can be used to estimate the variability of a statistic.

On the other hand, the central limit theorem (CLT) is a fundamental result in probability theory that states that the sum or average of a large number of random variables, each with finite mean and variance, will follow a normal distribution regardless of the underlying distribution of the individual variables. The CLT is a theoretical result that describes the behavior of distributions under certain conditions, while the bootstrap is a practical tool that can be used to estimate the variability of statistics based on observed data.

In practice, the bootstrap can be used to approximate the standard error of a statistic, which is a key component of many hypothesis tests and confidence interval calculations. By repeatedly resampling the observed data and computing the statistic of interest, we can estimate the sampling distribution of the statistic and use it to calculate standard errors, confidence intervals, and p-values.



So while the bootstrap and the central limit theorem are related concepts, they describe different things and serve different purposes.


