# Outline

1. Data
1. Probability
1. Distributions
1. Statistics
1. ANOVA
1. Regression
1. $\chi^2$-Squared Analysis

# Data

- **Central Tendency**: no "spread-out" description, no description of the shape.
  - mean $\rightarrow$ average
  - median $\rightarrow$ middle value
  - mode $\rightarrow$ most common value
- **Dispersion**:
  - range: $max - min$
  - variance (population is not always literally everything, it depends on the context):
    - **Sample**: $s^2 = \frac{\sum (x_i - \mu)^2}{N-1}$
    - **Population**: $\sigma^2 = \frac{\sum (x_i - \mu)^2}{N}$
  - Standard Deviation = $\sqrt{\sigma^2}$, useful because it has the same units of the main variable.
- **Quartiles**: Value that splits 25% of the data. If it is between two data points, use an average.
  - **Box Plots**:
    - The bulk is the IQR.
    - The line that divides the bulk is the median.
    - The whiskers are the min and max without outliers.
  - **Interquartile Ranges**: distance between q1 and q3.
    - **Fence and Outliers**:
      - This is determined by the data, not a prior definition!
      - Commonly, we fence at 1,5 x IQR beyond q1 and q3.
      - Outliers are drawn away from the whiskers.
- **Plotting**:
  - Correlation is not necessarily causation.
  - Covariance: $cov(X,Y) = \frac{1}{N} \sum^{N}_{i=1} (x_i - \bar{x})(y_i - \bar{y})$
  - **Pearson Correlation Coefficient**:
    - Standard correlation doesn't normalize the data, so it's difficult to know the degree of correlation for the specific dataset.
    - The normalized covariance or Pearson Correlation Coefficient:
    \[
    \rho_{XY} = \frac{cov(X,Y)}{\sigma_X \sigma_Y} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2} \sqrt{\sum (y_i - \bar{y})^2}}
    \]

# Probability

- **Permutations**:
  - $P_n = n!$
  - Out of a subset or **Arrangements**: $P_{n,r} = \frac{n!}{(n-r)!}$
  - With repetition: $n^r$
- **Combinations**: Unordered groups.
  - $C_{n,r} = \frac{n!}{r!(n-r)!}$
  - With **repetition**: $C_{n+r-1,r} = \frac{(n+r-1)!}{r!(n-1)!}$
- **Intersections, Unions and Complements**:
  - $A \cap B \Rightarrow AND$
  - $A \cup B \Rightarrow OR \Rightarrow P(A) + P(B) - P(A \cap B)$ (Not necessarily $A \cup B = U$)
  - $\bar{A} = U - A \Rightarrow 1 - P(A)$
- **Conditional Probability**:
  - Most Important Formula (more than Bayes's): $P(R_1 \cap R_2) = P(R_1) \cdot P(R_2|R_1)$
  - From the above formula, we can derive:
  \[
  \begin{align*}

    &P(A|B) = \frac{P(A \cap B)}{P(B)} \\

    &P(B|A) = \frac{P(A \cap B)}{P(A)} \\

    \Rightarrow &P(A|B) = \frac{P(B|A) P(A)}{P(B)}

  \end{align*}
  \]
  - **Example**: For a given text, the probability of a defective product is 0,002. The accuracy of the test is 0,99. What is the probability of a product being defective, given that it tested positive?
    \[
    \begin{align*}

      P(Def|Pos) &= \frac{P(Pos|Def)P(Def)}{P(Pos)} = \\
      &= \frac{0,99 \cdot 0,002}{P(Pos|Def)P(Def) + P(Pos|nDef)P(nDef)} = \\
      &= \frac{0,99 \cdot 0,002}{0,99 \cdot 0,002 + 0,01 \cdot 0,998} = 16,5\%

    \end{align*}
    \]
    - What if there was a second test?
      \[
      P(Def|Pos|Pos) = \frac{0,99 \cdot 0,165}{0,99 \cdot 0,165 + 0,01 \cdot 0,835} = 95,1\%
      \]

# Distributions

- **Discrete Distributions**:
  - **Uniform**
  - **Binomial**
    - Bernoulli Trials: 2 Possible outcomes with independent trials.
    - Probability of $x$ successes:
      \[
      \begin{align*}
        & P(x, n, p) = \binom{n}{x} p^x (1 - p)^{n-x} \\
        & \mu = np \\
        & \sigma^2 = np(1 - p)
      \end{align*}
      \]
    - In python:
      ```python
      from scipy.stats import binom
      binom.pmf(3,16,1/6)
      ```
  - **Poisson**
    - The difference between Bernoulli and Poisson is that Poisson takes the # of successes per unit of time (continous unit).
    \[
    \begin{align*}

      & \lambda = \frac{\# occurrences}{interval} = \mu = \sigma^2 \\

      & P(X = x) = \frac{\lambda^{x} e^{-\lambda}}{x!} \\

      & cdf(X) = P(X = x < N) = \sum^{N}_{i=0} \frac{\lambda^i e^{-\lambda}}{i!}

    \end{align*}
    \]
    - It also assumes that the probability during small time intervals is proportional to the entire length, e.g., $\lambda_{min} = \lambda_{hour}/60$.
- **Continuous Distributions**
  - Normal
    \[
    f(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{1}{2} \frac{(x-\mu)^2}{\sigma^2}}
    \]
    - Usually under its standard parameters: $N(0,1)$.
    - Standard Deviation Distances:
      - 1-sigma = 68,27%
      - 2-sigma = 95,45%
      - 3-sigma = 99,73%
    - **Z-Scores**: $Z = \frac{x - \mu}{\sigma}$
    - In python:
      ```python
      from scipy import stats
      stats.norm.cdf(z = 0.7)
      stats.norm.ppf(p = 0.95)
      ```
  - Exponential
  - Beta

# Statistics

- **Sampling**
  - **>30** is usually quite good.
  - **Selection Bias**:
    - The chosen samples don't quite reflect the population
      - Undercoverage
      - Self-selection
      - Healthy user
      - Survivorship (more helmets, more head injuries, but that's because more people are coming back alive.)
  - **Types**:
    - Random
    - Stratified Random
    - Cluster (less precise)
- **Central Limit Theorem (CLT)**:
  - **Mean value** will be normally distributed **even if** the population itself is not normally distributed.
  - The Z-score for the mean will be $Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}}$.
  - The standard error = $\sigma/\sqrt{n}$, or, in an approximation, $s_n/\sqrt{n}$.
  - If you wish to know how similar $F_n(x)$ and $\Phi(x)$ are, see the [Berry-Esséen Inequality](https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem).
- **Hypothesis Testing**:
  - Start with a **Null Hypothesis ($H_0$)**:
    - You do not *accept* a hypothesis, but **fail to reject it** or **reject it** (assume another mutually exclusive hypothesis).
    - $H_0$ should *usually* be the opposite of what you want to "prove".
    - We never *prove* a hypothesis.
    - The null hypothesis should contain an (in)equality ($<, =, >$).
      - For Z-scores:
        - $> \rightarrow left-tail$
        - $< \rightarrow right-tail$
        - $= \rightarrow two-tail$
    - **Level of Significance ($\alpha$)**: tails of the null hypothesis.
    - Testing **Means** vs **Proportions**:
    \[
    \begin{align*}

      & Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \\

      & Z = \frac{\hat{p} - p}{\sqrt{\frac{pq}{n}}} = \frac{\hat{p} - p}{\sqrt{\frac{p(1-p)}{n}}}

    \end{align*}
    \]
      - **P-value Test**:
        1. Take test statistic
        1. Use it to determine the P-value
        1. Compare the P-value to the level of significance $\alpha$
          - P-value is low, Null must go $\rightarrow$ reject $H_0$
          - P-value is high, Null must fly $\rightarrow$ fail to reject $H_0$
  - **Example 1**:
    - Servers with $\mu_0 = 3,125s$ and $\sigma_0 = 0,7s$ of load time.
    - Desired Confidence on Improvement: $99\% \rightarrow \alpha = 0,01$.
    - $n = 40$
    - New average $\rightarrow \mu^{\prime} = 2,875s$
    - $H_0: \mu \geq 3,125$ \rightarrow$ left-tail.
    - $Z = \frac{\mu^{\prime} - \mu}{\sqrt{\sigma}/n} = -2,259$
    - $\alpha = 0,01 \rightarrow Z_{\alpha} = -2,325 \rightarrow$ fail to reject $H_0$
      - or $P-value \rightarrow 0,0119 > 0,01 \rightarrow$ fail to reject $H_0$
      - We can safely say that, with 99% confidence, the new average is not an improvement.
  - **Example 2**:
    - $n = 400$
    - $58\% \ teenagers$
    - Are most customers teenagers?
    - $H_0: P \leq 0,5$
    - $Z = \frac{0,58 - 0,5}{\sqrt{\frac{0,5 (1 - 0,5)}{400}}} = \frac{0,08}{0,025} = 3,2$
    - $\alpha = 0,05 \rightarrow Z_{\alpha} = 1,645 \rightarrow$ right-tail
    - $3,2 > 1,645 \rightarrow$ reject $H_0$
      - With 95% confidence, most customers are indeed teenagers
  - **Type I and Type II Errors**:
    - **Type I**: False Negative - failing to reject a false $H_0$.
      - Saying to a man "you're pregnant".
    - **Type II**: False Positive - rejecting a true $H_0$
      - Saying to an obviously pregnant woman "you're not pregnant".
  - **Student's t-Distribution**:
    - The t-Distribution is similar to the normal distribution, but it has fatter tails and, consequently, lower levels at the mean.
      - If **$n \geq 30 \rightarrow t \approx Z$**, and that's why people often use $Z$ instead of $t$ when $n \geq 30$.
    - by William Sealy Gossett, a Guinness Brewer, with a pseudonym of Student.
      - Select the best barley from small samples when the standard deviation was unknown. Finally, we admit that **we don't know the underlying sample's mean or variance**.
        - t-Table with t-Statistics
        - t-Test determines if there is significant difference between two sets of data
        - Types of t-Tests:
          1. One sample t-Test ($\mu_n = \mu$?)
          1. Independent 2 sample t-Test ($\mu_1 = \mu_2$?)
          1. Dependent paired sample t-Test (e.g.: test scores before and after a prep course.)
    - **One Sample t-Test**
      - $t = \frac{\bar{x} - \mu}{s_n / \sqrt{n}}$, where $s_n$ is the sample's variance.
      - In order to continue, we do the following:
        1. Degrees of Freedom ($df$)
        1. Choose a significance level
        1. Look at the t-critical
        1. Compare with our t.
    - **Independent 2 Sample t-Test**
      - There are 3 variations:
        - equal sample sizes, equal variance
        - unequal sample sizes, equal variance
        - unequal or equal sample sizes, unequal variance (most common)
      - **Formulas**:
        \[
        \begin{align*}

          t &= \frac{signal}{noise} = \frac{difference \ in \ means}{sample \ variability} = \frac{|\overline{x_1} - \overline{x_2}|}{\sqrt{\frac{s_{1}^{2}}{n_1} + \frac{s_{2}^{2}}{n_2}}} \\

          df &= \frac{(\frac{s_{1}^{2}}{n_1} + \frac{s_{2}^{2}}{n_2})^2}{\frac{1}{n_1 - 1}(\frac{s_{1}^{2}}{n_1})^2 + \frac{1}{n_2 - 1}(\frac{s_{2}^{2}}{n_2})^2} \\ \\

          if \ s_1 &= s_2: \\ \\

          &df = n_1 + n_2 - 2 = (n_1 - 1) + (n_2 - 1)

        \end{align*}
        \]
      - **Example**:
        - Two plants, same car, which one to close?
        - $n_A = 10, \ \overline{x_A} = 1222, \ n_B = 10, \ \overline{x_B} = 1186$
          - Is A statistically similar to B? (If we get beyond the $t_{critical}$, we are different.)
          - $H_0: X_A \leq X_B \rightarrow$ one-tailed t-Test
        \[
        \begin{align*}

          &df = 10 + 10 - 2 = 18 \\

          &s_{A}^2 = 1248, \ s_{B}^2 = 1246 \\

          &t = \frac{|1222 - 1186|}{\sqrt{\frac{1248}{10} + \frac{1186}{10}}} = 2,28 \\

          &t_{critical, \ \alpha = 0,05} = 1,734 \\ \\

          \Rightarrow &t = 2,28 > 1,734 = t_{crit} \Rightarrow reject \ H_0

        \end{align*}
        \]
          - That is: Plant A does statistically, with 95% confidence, produce more than Plant B.

# ANOVA

- **Previously**: What is the probability that two samples come from a population with the same variance?
  - But what if you have 3 samples? You can make 3 pairings and t-Test them.
    - However, the overall confidence will drop: $0,95 \cdot 0,95 \cdot 0,95 = 0,857$
- **Analysis of Variance**, F-Distribution (not symmetrical)
  - The smaller the $\alpha$ (higher confidence), the bigger the $F_{critical}$, i.e., the more extreme the differences between the groups have to be.
  - **Variance**
    - between groups
    - within groups
    \[
    \begin{align*}

      & F = \frac{var \ between \ groups}{var \ within \ groups} = \frac{\frac{SSG}{df_{groups}}}{\frac{SSE}{df_{error}}} \\ \\

      & SSG = n_{samples \ in \ group} \cdot \sum (\mu_i - \mu_T)^2 \\

      & df_{groups} = n_{groups} - 1 \\

      & SSE = \sum (x_i - \mu_i)^2 \\

      & df_{error} = (n_{samples \ in \ group} - 1) \cdot n_{groups}

    \end{align*}
    \]
- **Example 1**:
  - Give different discounts to customers.
    - Does it make them pay early?
      - $H_0:$ discounts yield the same behavior (distribution)
  - 3 groups: 2%, 1% and 0% discount. Each sample is the number of days it took for the customer to pay with that discount.
    \[
    \begin{align*}

      & \sum (\mu_i - \mu_T)^2 = 14 \\

      & \sum (x_i - \mu_i)^2 = 198 \\ \\

      & F = \frac{\frac{14 \cdot 5}{3 - 1}}{\frac{198}{(5-1) \cdot 3}} = 2,121 \\

      & F_{critical} = 3,885 \\ \\

      \Rightarrow & F = 2,121 < 3,885 = F_{critical} \Rightarrow fail \ to \ reject \ H_0

    \end{align*}
    \]
- **Two-Way ANOVA**
  - More variables (blocks), e.g., 3 discount groups with different amounts of payments for different samples (blocks).
    - Now, rows are blocks.
    - You want to isolate and remove any variance contributed by the blocks to better understand the variance in the groups.
    \[
    \begin{align*}

      & SSError = SSTotal - SSGroups - SSBlocks \\

      & df_{groups} = n_groups - 1 \\

      & df_{error} = (n_{blocks} - 1)(n_{groups} - 1) \\

      & F = \frac{\frac{SSG}{df_{groups}}}{\frac{SSE}{df_{error}}}

    \end{align*}
    \]
  - **Example 2**:
    - $H_0:$ no statistical difference.
    \[
    \begin{align*}

      & SSG = n_{samples \ in \ group}\sum (\mu_{group} - \mu_T)^2 = 5 \cdot 14 = 70 \\

      & df_{groups} = 3 - 1 = 2 \\

      & SSB = n_{samples \ in \ block} \sum (\mu_{block} - \mu_{T})^2 = 3 \cdot 58 = 174 \\

      & SST = \sum (x_{i} - \mu_T)^2 = 268 \\

      & SSE = SST - SSG - SSB = 24 \\

      & df_{error} = (5-1)(3-1) = 8 \\

      & F = \frac{\frac{70}{2}}{\frac{24}{8}} = 11,67 \\ \\

      & \alpha = 0,05; \ df_{num} = 2; \ df_{denom} = 8 \Rightarrow F_{critical} = 4,46 \\ \\

      \Rightarrow & F_{critica} = 4,46 < 11,67 = F \Rightarrow reject \ H_0

    \end{align*}
    \]
    - Giving discounts now yields statistically different results.
      - Same dataset as in example 1, but now with blocks.
- **Two-Way ANOVA with Repetition**:
  - Now each block may have more than one sample.
  - We need another statistic to account for the interactions (SSI).
    - Sample Means (sm) are means on group-block-wise sample.
  - **Example 3**
    - Types of fertilizers A, B and C (groups)
    - Temperatures warm and cold (blocks)
    - $H_0:$ no statistical difference.
    \[
    \begin{align*}

      & SSB = n_{samples \ in \ block} \sum (\mu_{block} - \mu_{T})^2 = 9 \cdot 2 = 18 \\

      & SSG = n_{samples \ in \ group}\sum (\mu_{group} - \mu_T)^2 = 6 \cdot 2 = 12 \\

      & df_{groups} = 3 - 1 = 2 \\

      & df_{blocks} = 2 - 1 = 1 \\

      & SST = 164 \\

      & SSI = n_{items \ in \ sample} \sum_{sm} (\mu_{sm} - \mu_{block} - \mu_{group} + \mu_T)^2 = 3 \cdot 28 = 84 \\

      & SSE = SST - SSG - SSB - SSI = 50 \\

      & df_{error} = n_{blocks} \cdot n_{groups} \cdot (n_{items \ in \ sample} - 1) = 2 \cdot 3 \cdot (3 - 1) = 2 \\

      & F = \frac{12/2}{50/12} = 1,44 \\

      & \alpha = 0,05 \Rightarrow F_{critical} = 3,885 \\ \\

      \Rightarrow & F = 1,44 < 3,885 = F_{crit} \Rightarrow fail \ to \ reject \ H_0

    \end{align*}
    \]

# Regression

\[
\begin{align*}

  & \hat{y} = b_0 + b_1 x \\

  & b_1 = \rho_{xy} \frac{\sigma_{y}}{\sigma{x}} \\

  & b_0 = \bar{y} - b_1 \bar{x}

\end{align*}
\]
- where $\rho_{xy}$ is the Pearson Correlation Coefficient.
- **Limitations**
  - Anscombe Quartet (1973) illustrated the pitfalls of relying on pre calculation.
    - For the same regression line, you can have totally different underlying functions.
- **Multiple Regression**:
  \[
  \begin{align*}

    & \hat{y} = b_0 + b_1 x_1 + b_2 x_2 + \cdots \\

    & b_1 = \frac{\sum (x_2 - \bar{x}_2)^2 \sum (x_1 - \bar{x}_1)(y - \bar{y}) - \sum (x_1 - \bar{x}_1)(x_2 - \bar{x}_2) \sum (x_2 - \bar{x}_2)(y - \bar{y})}{\sum (x_1 - \bar{x}_1)^2 \sum (x_2 - \bar{x}_2)^2 - (\sum (x_1 - \bar{x}_1)(x_2 - \bar{x}_2))^2} \\

    & b2 = \cdots

  \end{align*}
  \]
  - Avoid using factors which don't have much correlation between them , they will just be noise.

# $\chi^2$-Analyis

- by Karl Pearson (1900): how much our observations diverge from the expected.
  - The trials have to be independent.
  - Either 0 or 1
- "The product of two independent gaussians is a chi-squared distribution":
  \[
  Q = \sum^{K}_{i=1} Z_{i}^2 \sim \chi^2(k)
  \]
  - where $K$ is the degrees of freedom.
- The basics with a simple example:
  - 18 flips of a coin, with 12 heads. Are 12 heads reasonable?
  - $H_0:$ 12 heads is reasonable.
  \[
  \begin{align*}

    & \chi^2 = \sum \frac{(O - E)^2}{E} = \frac{(12 - 9)^2}{9} + \frac{(6 - 9)^2}{9} = 2 \\

    & df_{coins} = 2 - 1 = 1 \\ \\

    \Rightarrow & \chi^2 = 2 < 3,841 = \chi^{2}_{critical} \Rightarrow fail \ to \ reject \ H_0

  \end{align*}
  \]
- **Example**:
  - Based on the following data, can we assume that servers fail at the same rate?
  - Calculating the Expected:
    - Use an average of the observed: $\sum x_i / n_{items} = 240/6 = 40$
  - Assumptions:
    - failures are independent
    - no "degrees of failure", either fail or not
  \[
  \begin{align*}

    & \chi^{2} = \sum \frac{(O - E)^2}{E} = 10 \\

    & \alpha = 0,05 \\

    & df = 6 - 1 = 5 \\

    & \chi^{2}_{crit} = 11,07 \\ \\

    \Rightarrow & \chi^2 = 10 < 11,07 = \chi^2_{crit} \Rightarrow fail \ to \ reject \ H_0

  \end{align*}
  \]
  - 95% confidence that they converge to what is expected.
