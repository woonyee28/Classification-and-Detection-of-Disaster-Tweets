# Classification and Detection of Disaster Tweets

## About
Natural Disasters have caused an average of 60,000 deaths worldwide. When Natural Disaster strike, many that have witnessed it would often report it on social media in real time which can be done through twitter or facebook. Many often seek news from social media as it is much faster than traditional media. Since people would report it on social media, there is a need for fast response from the rescue operators to respond to the disaster. However, there is currently no system in place to alert the rescue operators about a disaster that is posted on social media.

The goal of this project is to identify tweets that are deemed as a Disaster Tweet through the use of Machine Learning

In order to achieve the goals set out, we will need to:
* Find a suitable dataset
* Clean the dataset
* Find a suitable model for training
* Implement the idea (through a website)

![mini-project](https://user-images.githubusercontent.com/32679064/161420789-8ce35467-9ec5-4997-947d-efdf00348fc6.gif)

## Full Presentation
Presentation Video: [Click Here](https://www.youtube.com/watch?v=9YFjJz0AOWY&ab_channel=NgWoonYee)     
Full Code in Google Colab: [Click Here](https://colab.research.google.com/drive/1Ui0s3H-CxEHLlRc0PvCCcfNcxYIcEwzE?usp=sharing)     
For detailed walkthrough, please view the source code in order from:   

1. [Data Extraction and Data Cleaning](https://github.com/woonyee28/mini-project/blob/main/Data_Extraction_and_Data_Cleaning.ipynb)
2. [Data Visualization and Data Pre-processing](https://github.com/woonyee28/mini-project/blob/main/Data_Visualization_and_Data_Pre_processing.ipynb)
3. [Dense Network, LSTM and Bi-LSTM](https://github.com/woonyee28/mini-project/blob/main/Dense_Network%2C_LSTM_and_Bi_LSTM.ipynb)
4. [Comparison and Other Methods](https://github.com/woonyee28/mini-project/blob/main/Comparison_and_Other_Methods.ipynb)
5. [Website](https://github.com/woonyee28/mini-project/blob/main/index-streamlit.py)

## Dataset used
We used this [dataset](https://www.kaggle.com/competitions/nlp-getting-started/data) provided by Kaggle for our project

## Models used
- Dense Network
- Long Short-Term Memory (LSTM) Network 
- Bi-directional LSTM Network  

## Conclusion
- Between Dense Network, LSTM Model, and Bi-Directional LSTM, Dense Network has the highest accuracy.  
- Data overfitting would slightly reduce accuracy of the model  
- With close to 80% accuracy, our model did well on the classification most of the time.  

## Takeaways
- Data Cleaning
  - Using `regex` to remove unwanted characters
  - Fixed the imbalanced dataset to prevent inaccuracy of data  
- Data Visualisation 
  - Prepare and visualise our data using `wordcloud`  
- Data Pre-processing (Text Processing) 
  - Use of `tokenization`  
  - Use of `sequencing`  
  - Use of `padding` 
- Machine learning 
  - `Dense Network` using keras  
  - `Long Short-Term Memory (LSTM)` Network 
  - `Bi-directional LSTM` Network 
- Website 
  - Use of `Streamlit`
  - Deploy in both localhost and home server to test run

## References
- https://www.kaggle.com/competitions/nlp-getting-started/data  
- https://www.kaggle.com/datasets/phanttan/disastertweets-prepared  
- https://towardsdatascience.com/beginners-guide-for-data-cleaning-and-feature-extraction-in-nlp-756f311d8083  
- https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots  
- https://www.datacamp.com/community/tutorials/wordcloud-python  
- https://towardsdatascience.com/nlp-preparing-text-for-deep-learning-model-using-tensorflow2-461428138657  
- https://www.analyticsvidhya.com/blog/2021/06/nlp-sentiment-analysis/  
- https://analyticsindiamag.com/complete-guide-to-bidirectional-lstm-with-python-codes/  
- https://towardsdatascience.com/nlp-spam-detection-in-sms-text-data-using-deep-learning-b8632db85cc8  
- https://ourworldindata.org/natural-disasters


## Contributors
- `woonyee28` - Website, Implementation and Setup of Idea
- `Baby-McBabyFace` - Data Cleaning, Data Visualization, Data Pre-processing
- `keenlim` - Machine Learning Models, Comparison of data
