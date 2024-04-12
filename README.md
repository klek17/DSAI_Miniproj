# DSAI Mini-Project
For our mini project in the Introduction to Data Science and Artificial Intelligence module (SC1015), we performed analysis on the Anime Dataset with Reviews - MyAnimeList [dataset](https://www.kaggle.com/datasets/marlesson/myanimelist-dataset-animes-profiles-reviews) from Kaggle. This dataset contains informations about Anime (16k), Reviews (130k) and Profiles (47k) crawled from https://myanimelist.net/ at 05/01/20.

### Problem Definition
- Which genre of anime is the most popular and why?
- As an anime studio, how to tell if an anime is worth producing based off a synopsis


### Members (FCSD)
1. Kevin Lim Ern Kee
2. Shaun Cheong
3. Nigel Tan

### Files Included
1. animes.csv - contains list of anime, with title, title synonyms, genre, duration, rank, populatiry, score, airing date, episodes and many other important data about individual anime
2. reviews.csv - contains information about reviews users x animes and scores
3. profiles.csv - contains information about users who watch anime, namely username, birth date, gender, and favorite animes list
4. EDA.ipynb
    - Cleaning and preparation
    - Basic visualization
    - Exploratory data analysis
5. NLTK_synopsis.ipynb
    - Cleaning and preparation
    - Natural Language Processing: Natural Language Toolkit (NLTK)
6. BERT_synopsis.ipynb
    - Cleaning and preparation
    - Deep Learning: Bidirectional Encoder Representations from Transformers (BERT)


### EDA.ipynb Details
#### Cleaning and Preparation
   a. Merge profile, animes, reviews csv file into a single dataframe

   b. Remove duplicate rows which were created during merging

   c. Obtain a cleaned list of all genres


#### Basic Visualization
   a. Used word cloud and bar plot to display most common anime genre

   b. Barplot of average number of episodes, members and score per genre


#### Exploratory Data Analysis
   a. Show the breakdown of top 5 genres by gender

   b. Boxplot of members by genre

   c. Correlation of episodes, members and score

   
### NLTK_synopsis.ipynb Details
#### Cleaning and Preparation
   a. Read csv columns synopsis and genre into a dataframe

   b. Remove rows with no synopsis available

   c. Count the total number of genres

   d. Clean synopsis by removing apostrophes, removing non-alphabetic characters, removing extra spaces, ensuring that words are separated by single spaces and converts the text to lowercase

   
#### Natural Language Processing (NLTK)
*1. Removing stopwords*

    stop_words = set(stopwords.words('english'))
    stop_words.add('source')
    words = defaultdict(lambda: 0)
    def rem_stop(text):
        for i in text.split():
            new_text = []
            if i not in stop_words:
                new_text.append(i)
                words[i] += 1
        new_text = [i for i in text.split() if not i in stop_words]
        return ' '.join(new_text)
    
    df.loc[:, 'cleaned_syn'] = df.loc[:, 'cleaned_syn'].apply(rem_stop)

*2. Build a multilabel binarizer using genrelist*

    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(df['genrelist'])
    y = multilabel_binarizer.transform(df['genrelist'])

*3. Split train test data 80-20 split and initialise TF-IDF vectorizer*

    xtrain, xtest, ytrain, ytest = train_test_split(df['cleaned_syn'], y, test_size=0.2)
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    xtest_tfidf = tfidf_vectorizer.transform(xtest)

*4. Train the Logistic Regression classifier and predict on test set*

    model = LogisticRegression()
    classif = OneVsRestClassifier(model)
    classif.fit(xtrain_tfidf, ytrain)
    y_pred = classif.predict(xtest_tfidf)
    y_pred_prob = classif.predict_proba(xtest_tfidf)
    y_pred_new = (y_pred_prob >= 0.20).astype(int)


### BERT_synopsis.ipynb Details
#### Cleaning and Preparation
   a. Read csv columns synopsis and genre into a dataframe

   b. Remove rows with no synopsis available

   c. Count the total number of genres

   
#### Deep Learning (BERT)
*1. Loading BERT tokenizer and model*

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(all_genres))
    max_length = 512  # determines the maximum number of tokens allowed in a single input sequence

*2. Encode labels based on the length of genre_list using one-hot encoding*

    for synopsis, genre_list in zip(df['synopsis'], df['genre_list']):
    tokenized_text = tokenizer.encode_plus(synopsis, add_special_tokens=True, max_length=max_length, padding='max_length', truncation=True)
    tokenized_texts.append(tokenized_text)
    label = torch.zeros(len(all_genres))
    for genre in genre_list:
        label[all_genres.index(genre)] = 1
    labels.append(label)

*3. Split data into training and validation sets 80-20 split*

    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
    input_ids, attention_masks, labels, random_state=42, test_size=0.2=)

*4. Create DataLoader for training and validation sets*

    batch_size = 8  # determines how many samples are processed in each iteration of training
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_loader = DataLoader(val_data, batch_size=batch_size)

*5. Create optimizer and scheduler*

    from transformers import AdamW, get_linear_schedule_with_warmup
    optimizer = AdamW(model.parameters(),
                      lr=1e-5, 
                      eps=1e-8,
                      no_deprecation_warning=True)
    epochs = 5 # determines how many times the entire training dataset is passed forward and backward through the model during training
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_loader)*epochs)
*6. Batch training*

    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}', leave=False):
        input_ids, attention_masks, labels = batch
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        
    # Calculate average training loss for the epoch
    avg_train_loss = running_loss / len(train_loader)

*7. Validation phase*

    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    all_predicted_labels=[]
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}', leave=False):
            input_ids, attention_masks, labels = batch
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits
            
            # Compute loss
            loss = criterion(logits, labels)
            val_loss += loss.item()
            # Compute accuracy
            probabilities = torch.sigmoid(logits)
            predicted_labels = (probabilities > 0.5).to(torch.float32)
            correct_predictions += torch.sum(predicted_labels == labels).item()
            total_predictions += labels.size(0) * labels.size(1)
            all_predicted_labels.extend(predicted_labels.tolist())
    
### Conclusion

*Data Driven Insights*
- Top 5 genres for both males and females are identical; Comedy, Action, Fantasy, Adventure and Drama
- If the genre of anime is Slice-of-life, Shoujo, Comedy, Game or Martial-Arts, the funding required would likely be higher as they have the highest average number of episodes
- Animes in the Mystery, Harem, Vampire, Psychological, Thriller would likely draw in the most revenue through merchandise sales as they have the highest average members
- From the heatmap we can see that number of episodes, number of members and score do not any correlation except for a slight correlation between number of members and score


*Machine Learning Comparisons*
- NLTK model consistently produced low accuracy and decent F1 score however showed no room for further improvement as it was a pre-trained model and could not train itself on the data provided
- BERT model produced consistently higher accuracies but lower F1 scores. However with each epoch, a clear increase in both accuracy and accuracy can be observed, hence showing BERT would be highly useful with further training


### What we have learnt from this project?
- Natural Language Toolkit
- Training Bidirectional Encoder Representations from Transformers
- Analysis of metrics such as accuracy and F1 score
