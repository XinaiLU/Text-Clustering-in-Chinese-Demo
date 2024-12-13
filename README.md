## A Demo of Text Clustering in Chinese

### I. Introduction  
A small academic competition on an interdisciplinary track, the goal is to classify agricultural bloggers on a short video platform based on data crawled. The idea is to combine the titles of 10 videos from each blogger into a single text without spaces, preprocess this text (removing stopwords, etc.), segment the words, calculate text similarity, and finally perform clustering.

Many of the codes and ideas in the article are borrowed from the "Deep Learning and Text Clustering: A Comprehensive Introduction and Practical Guide" on CSDN blog, but the deep learning model training was not used. Only the text was clustered. Suggestions for better preprocessing methods and visualization techniques are welcome!

![image](https://github.com/user-attachments/assets/2ff07606-d15c-4321-ab43-698826f6456b)


### II. Text Clustering  
Text clustering is the process of grouping text data based on the similarity of their content. The overall approach given by Prof.G is roughly as follows:

1. **Data Preprocessing**: Preprocess the text data, including text cleaning, word segmentation, removing stopwords, stemming or lemmatization, and other operations. These operations help reduce noise and redundant information in the data, extracting valid features of the text data.

2. **Feature Representation**: Represent the text data in a form that can be processed by computers. Common text feature representation methods include Bag-of-Words, TF-IDF (Term Frequency-Inverse Document Frequency), etc. These methods transform text data into vector representations for subsequent calculations and analysis.

3. **Clustering Algorithm Selection**: Choose the appropriate clustering algorithm for text data clustering. Common text clustering algorithms include K-means, hierarchical clustering, density-based clustering, etc. Different clustering algorithms have different characteristics and applicable scenarios, and the appropriate algorithm should be selected according to the specific situation.

4. **Clustering Model Training**: Based on the selected clustering algorithm, train the clustering model on the preprocessed and feature-represented text data. The training process of the clustering model is to divide the data into several categories based on the features of the text data.

5. **Clustering Results Analysis**: Analyze and evaluate the clustering results to check if the text data in each category has a certain internal similarity and meets the expectations. This can be done using clustering result visualization, evaluation metrics (such as silhouette score, mutual information, etc.).

6. **Parameter Tuning**: Adjust the parameters of the clustering algorithm or choose different algorithms based on the analysis and evaluation of the clustering results. Retrain the model until satisfactory clustering results are obtained.

### III. Project Processing  
**(1) Dataset Preparation**  
The dataset was crawled by a teammate from a data analysis website of a short video platform, and the label structure is as follows:


| Field Number | Field Name                            | Description                                   |
|--------------|---------------------------------------|-----------------------------------------------|
| 1            | **Sequence Number**                  | Unique ID for each blogger                   |
| 2            | **Blogger Name**                     | Name of the blogger                          |
| 3            | **Gender**                           | Gender of the blogger                        |
| 4            | **Region**                           | Blogger's region                             |
| 5            | **Age**                              | Age of the blogger                           |
| 6            | **MCN Institution**                  | Associated MCN institution                   |
| 7            | **Certification Information**        | Certification details                        |
| 8            | **Influencer Profile**               | Blogger's bio or profile description         |
| 9            | **Total Fans**                       | Total number of fans                         |
| 10           | **Fan Club Size**                    | Size of the fan club                         |
| 11           | **Sales Reputation**                 | Reputation for selling products              |
| 12           | **Livestream Sales Power**           | Influence in livestream sales                |
| 13           | **Video Sales Power**                | Influence in video sales                     |
| 14           | **Fan Size**                         | Fan size category                            |
| 15           | **Main Category 1**                  | Primary content category                     |
| 16           | **Main Category 2**                  | Secondary content category                   |
| 17           | **Sales Level**                      | Overall sales level                          |
| 18           | **Sales Information 1**              | First piece of sales information             |
| 19           | **Sales Information 2**              | Second piece of sales information            |
| 20           | **Livestream Sessions (30 Days)**    | Number of livestream sessions in 30 days     |
| 21           | **Average Livestream Duration (30 Days)** | Average duration of livestreams in 30 days |
| 22           | **Total Livestream Sales (30 Days)** | Total number of livestream sales in 30 days  |
| 23           | **Total Livestream Revenue (30 Days)** | Total livestream revenue in 30 days        |
| 24           | **Video Count (30 Days)**            | Number of videos in 30 days                  |
| 25           | **Average Video Duration (30 Days)** | Average duration of videos in 30 days        |
| 26           | **Total Video Sales (30 Days)**      | Total number of video sales in 30 days       |
| 27           | **Total Video Revenue (30 Days)**    | Total revenue from video sales in 30 days    |
| 28           | **Provincial Level Livestream Fans** | Livestream fans at the provincial level      |
| 29           | **City Level Livestream Fans**       | Livestream fans at the city level            |
| 30           | **Provincial Level Video Fans**      | Video fans at the provincial level           |
| 31           | **City Level Video Fans**            | Video fans at the city level                 |
| 32–42        | **Products 1–11**                    | Information about promoted products          |
| 43–50        | **Videos 1–8**                       | Titles of the blogger's videos               |



**(2) Import Packages**  
```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
```

**(3) Preprocessing**  

**Merge Video Columns, Remove Bloggers Without Video Information, and Remove Spaces from Video Titles**

```python
import pandas as pd

# Read the original CSV file
df = pd.read_csv('data1.csv')

# Keep only the first column of blogger names, and merge video columns 1-8 into the second column
df['视频合并'] = df[['视频1', '视频2', '视频3', '视频4', '视频5', '视频6', '视频7', '视频8']].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Keep only the blogger names and the merged video column
df_new = df[['博主名称', '视频合并']]

# Remove bloggers without video information
df_new.loc[:, '视频合并'] = df_new['视频合并'].str.replace(' ','')
df_new = df_new.dropna(subset = ['视频合并'], how = 'all')

# Remove rows with empty second column
df_new = df_new[df_new['视频合并'].apply(lambda x: len(x) > 0)]
```

**Remove Potentially Influencing Keywords**  
In this step, agricultural-related keywords were removed since all bloggers are in the agricultural field. If not removed, the similarity between different bloggers' video titles might become too high.

```python
# Text Preprocessing: Remove keywords like '农业', '三农', etc.
df_new.loc[:, '视频合并'] = df_new['视频合并'].str.replace('三农','')
df_new.loc[:, '视频合并'] = df_new['视频合并'].str.replace('农村','')
df_new.loc[:, '视频合并'] = df_new['视频合并'].str.replace('农业','')
df_new.loc[:, '视频合并'] = df_new['视频合并'].str.replace('乡村','')
df_new.loc[:, '视频合并'] = df_new['视频合并'].str.replace('农','')

print(df_new['视频合并'])

df_new['视频合并'].to_csv('视频合并.csv')
```

**Remove Stopwords**  
Here, I used a comprehensive Chinese stopword list, compiled from various sources, to filter out stopwords.

```python
import jieba
from zhon.hanzi import punctuation

# Read the stopwords file
stopwords_path = '停顿词.txt'  # Path to the stopwords file
stopwords = set()
with open(stopwords_path, 'r', encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip())

def chinese_word_cut(text):
    # Use jieba to segment Chinese text and filter out stopwords and punctuation
    words = jieba.cut(text)
    return ' '.join([word for word in words if word not in stopwords and word not in punctuation])

# Perform Chinese word segmentation and cleaning
df_new['视频合并'] = df_new['视频合并'].apply(chinese_word_cut)
```

Below is a stopwords file demo that can be copied and used in your project. You can get the full vision in another .txt file.

```
!
"
#
$
%
&
'
(
)
*
+
,
-
--
.
..
...
exp
sub
sup
|
}
~
~~~~
·
‘
’
’‘
“
”
→
∈［
∪φ∈
≈
①
②
②ｃ
③
③］
④
⑤
⑥
⑦
⑧
⑨
⑩
──
■
▲
　
、
。
〈
〉
《
》
》），
」
『
』
【
】
〔
〕
〕〔
㈧
一
一.
一一
一下
一个
一些
一何
一切
一则
一则通过
一天
一定
一方面
一旦
一时
一来
一样
一次
一片
一番
一直
... ... (you can get in another .txt file)
```

### IV. Text Representation and Dimensionality Reduction

#### 1. Text Representation
We used the `CountVectorizer` class to transform text data into a Bag-of-Words representation. `CountVectorizer` converts text data into a document-term frequency matrix, where each row represents a document, each column represents a word, and the elements indicate the frequency of the word in the corresponding document. The parameter `max_features=100000` restricts the vocabulary to the top 100,000 most frequent words as features.

#### 2. Dimensionality Reduction
We applied Singular Value Decomposition (SVD) to reduce the dimensionality of the document-term matrix. Specifically, we used the `TruncatedSVD` class to perform truncated SVD, reducing the matrix to 500 dimensions. Dimensionality reduction aims to extract the primary information in the data, simplifying subsequent text analysis and processing.

```python
# Text Representation
vectorizer = CountVectorizer(max_features=100000)
X = vectorizer.fit_transform(df_new['视频合并'])

dense_matrix = X.toarray()

# Dimensionality Reduction
svd = TruncatedSVD(n_components=500, n_iter=10, random_state=42)
X = svd.fit_transform(X)
```

Finally, the variable `X` stores the reduced text representation as a 2D array, where each row corresponds to the low-dimensional representation of a document.

![image](https://github.com/user-attachments/assets/cbb4391e-2dea-4c5c-8108-b21ca17bf356)

### V. Clustering

```python
kmeans = KMeans(n_clusters=8)
y = kmeans.fit_predict(X)

# Output clustering results
print(y)
len(y)
```

![image](https://github.com/user-attachments/assets/097d528f-52cd-4c4b-a831-175b0fd1764e)


### VI. Post-Processing

#### 1. Basic Information Output
Attach the cluster labels to the bloggers' information to facilitate output by category.

```python
df_new.insert(1, '类别', y)
df_new.to_csv('类别')

# Assuming 'category' contains cluster labels
for category, group in df_new.groupby('类别'):
    with open(f'category_{category}.csv', 'w', encoding='utf-8') as f:
        f.write(f'Category {category}\n')
        for index, row in group.iterrows():
            row_str = ','.join([str(val) for val in row.values]) + '\n'
            f.write(row_str)
```

After processing, the output is grouped by category (e.g., bloggers in Category 0 sell fruits, haha 😄).

![image](https://github.com/user-attachments/assets/9070e31d-e53a-4ef5-81ec-87bacda99eda)

- Bloggers specializing in durians.

![image](https://github.com/user-attachments/assets/f9536a2d-44a7-493a-a3b5-814cf0598c3d)

- Bloggers specializing in tea (though there's only one).
- 
![image](https://github.com/user-attachments/assets/e0e02260-2a95-4152-892b-08455cf9d16b)


#### 2. Visualization (Pending Improvements)

**Scatter Plot 1:**
Reduce the 100,000 features to 2 dimensions using PCA. However, this visualization doesn't seem very meaningful (compared to an earlier version that constrained the output range to prevent a single point being displayed far away while others are clustered below).

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Reduce features X for visualization
from sklearn.decomposition import TruncatedSVD
X_reduced = TruncatedSVD(n_components=2, random_state=42).fit_transform(X)

# Create a scatter plot with different colors for different clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette="viridis", s=50, alpha=0.8)
plt.title("K-Means Clustering Results")
plt.xlabel("Reduced Feature 1")
plt.ylabel("Reduced Feature 2")
plt.show()
```
![image](https://github.com/user-attachments/assets/4119ce0e-5704-416c-a8df-05e00672e0cb)

**Scatter Plot 2:**
Combine PCA with pairwise scatter plots of features by reducing 100,000 features to 100 and visualizing these smaller PCA features.

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD

# Reduce features X for visualization
X_reduced = TruncatedSVD(n_components=100, random_state=42).fit_transform(X)

# Create a scatter plot with different colors for different clusters
fig, axs = plt.subplots(20, 5, figsize=(15, 60))
axs = axs.flatten()

for i in range(100):
    sns.scatterplot(x=X_reduced[:, i], y=X_reduced[:, (i + 1) % 100], hue=y, palette="viridis", ax=axs[i], s=50, alpha=0.8)
    axs[i].set_xlabel(f'Feature {i}')
    axs[i].set_ylabel(f'Feature {(i + 1) % 100}')
    axs[i].legend()
    axs[i].set_xlim(-5, 5)
    axs[i].set_ylim(-5, 5)

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/16d6e57d-7f57-45b7-9f40-7f0fed5582ac)

It seems that the first few features differentiate the text more clearly, so I further split the first few features without PCA for pairwise scatter analysis.

**Scatter Plot 3:**
Following the above idea:

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Create KMeans model and fit the data
kmeans = KMeans(n_clusters=8, random_state=42)
clusters = kmeans.fit_predict(X[:, :10])

# Create a scatter plot with different colors for different clusters
fig, axs = plt.subplots(20, 5, figsize=(15, 60))
axs = axs.flatten()

for i in range(100):
    sns.scatterplot(x=X[:, i], y=X[:, (i + 1) % 100], hue=clusters, palette="viridis", ax=axs[i], s=50, alpha=0.8)
    axs[i].set_xlabel(f'Feature {i}')
    axs[i].set_ylabel(f'Feature {(i + 1) % 100}')
    axs[i].legend()
    axs[i].set_xlim(-15, 15)
    axs[i].set_ylim(-15, 15)

plt.tight_layout()
plt.show()
```
![image](https://github.com/user-attachments/assets/f66ec510-0c63-44e6-a3dc-030a0b700c3c)

#### Heatmap 1 (No Progress)

```python
import numpy as np
import seaborn as sns

# Assume kmeans is your KMeans model object, X is the feature representation
# Calculate the correlation matrix of features
correlation_matrix = np.corrcoef(X, rowvar=False)

# Create an array containing cluster labels
cluster_labels = kmeans.labels_

# Merge cluster labels with feature correlation matrix
merged_array = np.column_stack((cluster_labels, X))

# Calculate the mean correlation of each cluster
cluster_means = []
for cluster in np.unique(cluster_labels):
    cluster_data = merged_array[merged_array[:, 0] == cluster][:, 1:]
    cluster_means.append(np.mean(cluster_data, axis=0))

cluster_means = np.array(cluster_means)

# Draw the cluster heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(np.corrcoef(cluster_means, rowvar=False), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Cluster Correlation Heatmap')
plt.show()
```
![image](https://github.com/user-attachments/assets/98780c44-8e7d-43d8-a84b-1fd408a6578c)

This visualization doesn't show much progress either 😅.

---

### VII. Acknowledgments and Conclusion

Special thanks to my lovely teammates from the SN Experiment Class: Yaya, Zihan, and Tiezhu (in no particular order). 
 
Also, thanks to my equally lovely roommate Turing Ran for providing immensely strong and professional technical support for the second-round visualization updates.  

Wishing everyone success, happiness, and financial freedom!  
Singles, may you find love soon! 😊
