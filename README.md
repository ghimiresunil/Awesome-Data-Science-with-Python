# Awesome-Data-Science-with-Python
![0-02-03-38ae5bde119a673574e6e60c454e89f5b7fdda92d1e4e75b724444e9c5e84c09_1f8c7d07369fb7](https://user-images.githubusercontent.com/40186859/124362250-8f546080-dc53-11eb-9cbe-9986f5677102.jpg)

# 01. Basic Introduction
Great resources for learning data science with Python including tutorials, code snippets, blog pieces, and lectures, as well as libraries.

## 1.1. Environment and Jupyter 

- [General Jupyter Tricks](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)
- [Fix Jupyter Notebook](https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/)
- Python Debugger (PDB)
  -  [Blog Post](https://www.blog.pythonlibrary.org/2018/10/17/jupyter-notebook-debugging/)
  -  [Video](https://www.youtube.com/watch?v=Z0ssNAbe81M&t=1h44m15s)
  -  [Cheatsheet](https://nblock.org/2011/11/15/pdb-cheatsheet/)
- [cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science) - Data Science Project Templates.
- [nteract](https://nteract.io/) - Open Jupyter Notebooks with doubleclick.
- [papermill](https://github.com/nteract/papermill) - Parameterize and execute Jupyter notebooks([tutorial](https://pbpython.com/papermil-rclone-report-1.html)).
- [nbdime](https://github.com/jupyter/nbdime) - Different two notebook files, Alternative GitHub App: [ReviewNB](https://www.reviewnb.com/).
- [RISE](https://github.com/damianavila/RISE) - Turn Jupyter notebooks files into presentations.
- [qgrid](https://github.com/quantopian/qgrid) - Pandas `DataFrame` sorting.
- [pivottablejs](https://github.com/nicolaskruchten/jupyter_pivottablejs) - Drag n drop Pivot Tables and Charts for jupyter notebooks.
- [itables](https://github.com/mwouts/itables) - Interactive tables in Jupyter.
- [jupyter-datatables](https://github.com/CermakM/jupyter-datatables) - Interactive tables in Jupyter.
- [debugger](https://blog.jupyter.org/a-visual-debugger-for-jupyter-914e61716559) - Visual debugger for Jupyter. 
- [nbcommands](https://github.com/vinayak-mehta/nbcommands) - View and search notebooks from terminal.
- [handcalcs](https://github.com/connorferster/handcalcs) - More convenient way of writing mathematical equations in Jupyter.

## 1.2. Core 

- [NumPy](https://numpy.org/) - Multidimensional Arrays
- [Pandas](https://pandas.pydata.org/) - Data structures built on top of NumPy library
- [pandas_summary](https://github.com/mouradmourafiq/pandas-summary) - Basic statistics using `DataFrameSummary(df).summary()`.
- [pandas_profiling](https://github.com/pandas-profiling/pandas-profiling) - Descriptive statistics using `ProfileReport`.
- [Matplotlib](https://matplotlib.org/) - Visualization library.
- [Seaborn](https://seaborn.pydata.org/) - Data visualization library based on matplotlib.
- [scikit-learn](https://scikit-learn.org/stable/) - Core Machine Learning library.
- [sklearn_pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) - Helpful `DataFrameMapper` class.
- [missingno](https://github.com/ResidentMario/missingno) - Missing data visualization.
- [rainbow-csv](https://marketplace.visualstudio.com/items?itemName=mechatroner.rainbow-csv) - Plugin to display .csv files with nice colors.

## 1.3. Pandas Tricks, Additions, and Alternatives

- [Top 5 Best Pandas Tricks](https://towardsdatascience.com/5-lesser-known-pandas-tricks-e8ab1dd21431)
- [df.pipe](https://www.youtube.com/watch?v=yXGCKqo5cEY&ab_channel=PyData) -  Function to improve code readability
- [pandasvault](https://github.com/firmai/pandasvault) - Large collection of pandas tricks.
- [vaex](https://github.com/vaexio/vaex) - Out-of-Core DataFrames.
- [modin](https://github.com/modin-project/modin) - Parallelization library for faster pandas `DataFrame`.
- [xarray](https://github.com/pydata/xarray/) - Extends pandas to n-dimensional arrays.
- [pandarallel](https://github.com/nalepae/pandarallel) - Parallelize pandas operations.
- [pandapy](https://github.com/firmai/pandapy) - Additional features for pandas.
- [pandas-log](https://github.com/eyaltrabelsi/pandas-log) - Find business logic issues and performance issues in pandas.
- [pandas_flavor](https://github.com/Zsailer/pandas_flavor) - Write custom accessors like `.str` and `.dt`.
- [swifter](https://github.com/jmcarpenter2/swifter) - Apply any function to a pandas dataframe faster.

## 1.4. Helpful

- [tqdm](https://github.com/tqdm/tqdm) - Progress bars for for-loops. Also supports [pandas apply()](https://stackoverflow.com/a/34365537/1820480).
- [icecream](https://github.com/gruns/icecream) - Simple debugging output.
- [pyprojroot](https://github.com/chendaniely/pyprojroot) - Helpful `here()` command from R.
- [intake](https://github.com/intake/intake) - Loading datasets made easier, [talk](https://www.youtube.com/watch?v=s7Ww5-vD2Os&t=33m40s).
- [loguru](https://github.com/Delgan/loguru) - Python logging.

## 1.5. Extraction
- [camelot](https://github.com/socialcopsdev/camelot) - Extract text from PDF.
- [textract](https://github.com/deanmalmgren/textract) - Extract text from any document.

## 1.6. Big Data
- [spark](https://docs.databricks.com/spark/latest/dataframes-datasets/introduction-to-dataframes-python.html#work-with-dataframes) - `DataFrame` for big data
  -   [cheatsheet](https://gist.github.com/crawles/b47e23da8218af0b9bd9d47f5242d189)
  -   [tutorial](https://github.com/ericxiao251/spark-syntax).
- [sparkit-learn](https://github.com/lensacom/sparkit-learn)
  - [spark-deep-learning](https://github.com/databricks/spark-deep-learning) - ML frameworks for spark.
- [koalas](https://github.com/databricks/koalas) - Pandas API on Apache Spark.
- [dask](https://github.com/dask/dask)
  - [dask-ml](http://ml.dask.org/) - Pandas `DataFrame` for big data and machine learning library
  - [resources](https://matthewrocklin.com/blog//work/2018/07/17/dask-dev), [talk1](https://www.youtube.com/watch?v=ccfsbuqsjgI)
  - [talk2](https://www.youtube.com/watch?v=RA_2qdipVng)
  - [notebooks](https://github.com/dask/dask-ec2/tree/master/notebooks)
  - [videos](https://www.youtube.com/user/mdrocklin).
  - [dask-gateway](https://github.com/jcrist/dask-gateway) - Managing dask clusters.

## 1.7. Command line tools, CSV
- [ni](https://github.com/spencertipping/ni) - Command line tool for big data.
- [xsv](https://github.com/BurntSushi/xsv) - Command line tool for indexing, slicing, analyzing, splitting and joining CSV files.
- [csvkit](https://csvkit.readthedocs.io/en/1.0.3/) - Another command line tool for CSV files.
- [csvsort](https://pypi.org/project/csvsort/) - Sort large csv files.
- [tsv-utils](https://github.com/eBay/tsv-utils) - Tools for working with CSV files by ebay.
- [cheat](https://github.com/cheat/cheat) - Make cheatsheets for command line commands.

# 02. Classical Statistics

## 2.1. Statistical Tests and Packages

- [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-tests) - Statistical tests.
- [pingouin](https://github.com/raphaelvallat/pingouin) - Statistical tests.
- [researchpy](https://github.com/researchpy/researchpy) - Helpful `summary_cont()` function for summary statistics (Table 1).
- [scikit-posthocs](https://github.com/maximtrp/scikit-posthocs) - Statistical post-hoc tests for pairwise multiple comparisons.
- [Bland-Altman Plot](http://www.statsmodels.org/dev/generated/statsmodels.graphics.agreement.mean_diff_plot.html) - Plot for agreement between two methods of measurement.
- [ANOVA](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
  -  Tutorials: [One-way](https://pythonfordatascience.org/anova-python/)
  -  [Two-way](https://pythonfordatascience.org/anova-2-way-n-way/)
  -  [Type 1,2,3 explained](https://mcfromnz.wordpress.com/2011/03/02/anova-type-iiiiii-ss-explained/).

## 2.2. Visualizations
- [Correlation](https://rpsychologist.com/d3/correlation/)
- [Null Hypothesis Significance Testing (NHST) and Sample Size Calculation](https://rpsychologist.com/d3/NHST/)
- [Cohen's d](https://rpsychologist.com/d3/cohend/)
- [Confidence Interval](https://rpsychologist.com/d3/CI/)
- [Equivalence, non-inferiority and superiority testing](https://rpsychologist.com/d3/equivalence/)
- [Bayesian two-sample t test](https://rpsychologist.com/d3/bayes/)
- [Distribution of p-values when comparing two groups](https://rpsychologist.com/d3/pdist/)
- [Understanding the t-distribution and its normal approximation](https://rpsychologist.com/d3/tdist/)

## 2.3. Talks
- [Inverse Propensity Weighting](https://www.youtube.com/watch?v=SUq0shKLPPs)
- [Dealing with Selection Bias By Propensity Based Feature Selection](https://www.youtube.com/watch?reload=9&v=3ZWCKr0vDtc)
- [Talk](https://www.youtube.com/watch?v=68ABAU_V8qI)

## 2.4. Texts
- [Greenland - Statistical tests, P values, confidence intervals, and power: a guide to misinterpretations](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4877414/)
- [Lindeløv - Common statistical tests are linear models](https://lindeloev.github.io/tests-as-linear/)
- [Chatruc - The Central Limit Theorem and its misuse](https://lambdaclass.com/data_etudes/central_limit_theorem_misuse/)
- [Al-Saleh - Properties of the Standard Deviation that are Rarely Mentioned in Classrooms](http://www.stat.tugraz.at/AJS/ausg093/093Al-Saleh.pdf)
- [Cook - Estimating the chances of something that hasn’t happened yet](https://www.johndcook.com/blog/2010/03/30/statistical-rule-of-three/)
- [Wainer - The Most Dangerous Equation](http://www-stat.wharton.upenn.edu/~hwainer/Readings/Most%20Dangerous%20eqn.pdf)   
- [Gigerenzer - The Bias Bias in Behavioral Economics](https://www.nowpublishers.com/article/Details/RBE-0092)  

## 2.5. Frameworks
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - General machine learning framework.  
- [h2o](https://github.com/h2oai/h2o-3) - Machine learning framework.  
- [caffe](https://github.com/BVLC/caffe) - Deep learning framework
  -  [pretrained models](https://github.com/BVLC/caffe/wiki/Model-Zoo).  
- [mxnet](https://github.com/apache/incubator-mxnet) - Deep learning framework
  - [book](https://d2l.ai/index.html).  
  
## 2.6. Exploration and Cleaning
- [pyemd](https://github.com/wmayner/pyemd) - Earth Mover's Distance, similarity between histograms.  
- [Kaggler](https://github.com/jeongyoonlee/Kaggler) - Utility functions (`OneHotEncoder(min_obs=100)`)  
- [tspreprocess](https://github.com/MaxBenChrist/tspreprocess) - Time series preprocessing: Denoising, Compression, Resampling.  
- [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) - Resampling for imbalanced datasets.
- [fancyimpute](https://github.com/iskandr/fancyimpute) - Matrix completion and imputation algorithms.
- [impyute](https://github.com/eltonlaw/impyute) - Imputations.
- [janitor](https://pyjanitor.readthedocs.io/) - Clean messy column names.
- [littleballoffur](https://github.com/benedekrozemberczki/littleballoffur) - Sampling from graphs.
- [Checklist](https://github.com/r0f1/ml_checklist). 

## 2.7. Train / Test Split
- [iterative-stratification](https://github.com/trent-b/iterative-stratification) - Stratification of multilabel data.

## 2.8. Feature Engineering
- [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) - Pipeline
  - [Examples](https://github.com/jem1031/pandas-pipelines-custom-transformers).
- [pdpipe](https://github.com/shaypal5/pdpipe) - Pipelines for DataFrames.
- [scikit-lego](https://github.com/koaning/scikit-lego) - Custom transformers for pipelines.
- [skoot](https://github.com/tgsmith61591/skoot) - Pipeline helper functions.
- [categorical-encoding](https://github.com/scikit-learn-contrib/categorical-encoding) - Categorical encoding of variables
  - [vtreat (R package)](https://cran.r-project.org/web/packages/vtreat/vignettes/vtreat.html).
- [dirty_cat](https://github.com/dirty-cat/dirty_cat) - Encoding dirty categorical variables.
- [patsy](https://github.com/pydata/patsy/) - R-like syntax for statistical models.
- [mlxtend](https://rasbt.github.io/mlxtend/user_guide/feature_extraction/LinearDiscriminantAnalysis/) - LDA.
- [featuretools](https://github.com/Featuretools/featuretools) - Automated feature engineering
  -  [example](https://github.com/WillKoehrsen/automated-feature-engineering/blob/master/walk_through/Automated_Feature_Engineering.ipynb).
