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

## Models used
- Dense Network
- Long Short-Term Memory (LSTM) Network 
- Bi-directional LSTM Network  

## Conclusion
- Between Dense Network, LSTM Model, and Bi-Directional LSTM, Dense Network has the highest accuracy.  
- Data overfitting would reduce accuracy of the model  
- With close to 80% accuracy, our model did well on the classification most of the time.  
- Using real life example tweets, our model successfully classified tweets, with 1 false positive out of 14 tweets  

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
  - [PLACEHOLDER]

## References
- https://www.kaggle.com/competitions/nlp-getting-started/data  
- https://towardsdatascience.com/beginners-guide-for-data-cleaning-and-feature-extraction-in-nlp-756f311d8083  
- https://stackoverflow.com/questions/25447700/annotate-bars-with-values-on-pandas-bar-plots  
- https://www.datacamp.com/community/tutorials/wordcloud-python  
- https://towardsdatascience.com/nlp-preparing-text-for-deep-learning-model-using-tensorflow2-461428138657  
- https://www.analyticsvidhya.com/blog/2021/06/nlp-sentiment-analysis/  
- https://analyticsindiamag.com/complete-guide-to-bidirectional-lstm-with-python-codes/  
- https://towardsdatascience.com/nlp-spam-detection-in-sms-text-data-using-deep-learning-b8632db85cc8  

## Contributors
- `woonyee28` - Website, Implementation of idea
- `Baby-McBabyFace` - Data Cleaning, Data Visualization, Data Pre-processing
- `keenlim` - Machine Learning Models, Comparison
