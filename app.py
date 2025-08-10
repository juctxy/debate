import pandas as pd
import numpy as np
import nltk
import uuid
import os
import joblib
import gradio as gr
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

class DebateAnalyzer:
    """A class to analyze debate arguments and predict winners using machine learning."""
    
    def __init__(self, model_type='random_forest', is_speaker_level=False):
        """Initialize the DebateAnalyzer with specified model type and analysis level."""
        self.model_type = model_type
        self.is_speaker_level = is_speaker_level
        self.model = None
        self.calibrated_model = None
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.scaler = StandardScaler()
        self.fitted_vectorizer = None
        self.fitted_scaler = None
        self.feature_names = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def get_text_embeddings(self, text, max_features=50):
        """Generate text embeddings using TF-IDF and statistical features."""
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
        
        features = [
            len(filtered_tokens),  # Word count
            len(set(filtered_tokens)),  # Unique words
            len(text),  # Character count
            len(sent_tokenize(text)),  # Sentence count
        ]
        
        # Word length statistics
        word_lengths = [len(word) for word in filtered_tokens]
        features.extend([np.mean(word_lengths), np.std(word_lengths), 
                        max(word_lengths), min(word_lengths)] if word_lengths else [0, 0, 0, 0])
        
        # Character frequency features
        char_freq = {char: text.lower().count(char) for char in text.lower() if char.isalpha()}
        total_chars = sum(char_freq.values())
        common_chars = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r', 'd']
        features.extend([char_freq.get(char, 0) / total_chars if total_chars > 0 else 0 
                        for char in common_chars])
        
        # Punctuation features
        features.extend([
            text.count(p) / len(text) if len(text) > 0 else 0 
            for p in ['.', ',', '!', '?']
        ])
        
        # POS-like features
        features.extend([
            sum(1 for word in filtered_tokens if word.endswith(suffix)) / len(filtered_tokens) 
            if filtered_tokens else 0
            for suffix in ['ing', 'ed', 'ly']
        ])
        
        # Sentiment features
        blob = TextBlob(text)
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        features.extend([
            blob.sentiment.polarity,
            blob.sentiment.subjectivity,
            sentiment_scores['compound'],
            sentiment_scores['pos'],
            sentiment_scores['neu'],
            sentiment_scores['neg']
        ])
        
        # Pad or truncate to max_features
        return np.array(features[:max_features] + [0] * (max_features - len(features)))

    def train(self, X, y):
        """Train the model with resampled data using SMOTE."""
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        if self.model_type == 'xgboost':
            self.model = XGBClassifier(random_state=42)
            params = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
            grid = GridSearchCV(self.model, params, cv=5)
            grid.fit(X_resampled, y_resampled)
            self.model = grid.best_estimator_
        elif self.model_type == 'ensemble':
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            self.model = VotingClassifier(estimators=[('rf', rf), ('gb', gb), ('lr', lr)], voting='soft')
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=200, random_state=49, 
                                             class_weight='balanced', max_depth=10)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        else:
            raise ValueError("Unsupported model type")
        
        self.model.fit(X_resampled, y_resampled)
        self.calibrated_model = CalibratedClassifierCV(self.model, method='sigmoid', cv=3)
        self.calibrated_model.fit(X_resampled, y_resampled)

    def prepare_features(self, text, is_for_position, opponent_text=None):
        """Prepare features for prediction at segment or speaker level."""
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
        
        word_count = len(filtered_tokens)
        unique_words = len(set(filtered_tokens))
        evidence_words = sum(1 for word in filtered_tokens if word in [
            'affidavit', 'analysis', 'anecdotal evidence', 'best evidence', 'case', 'citation', 
            'circumstantial evidence', 'clue', 'confirmation', 'corroboration', 'data', 'datum', 
            'demonstration', 'deposition', 'detail', 'direct evidence', 'documentation', 
            'empirical evidence', 'evidence', 'examination', 'example', 'exhibit', 'experiment', 
            'exploration', 'fact', 'figure', 'finding', 'illustration', 'indication', 
            'indirect evidence', 'information', 'inquiry', 'investigation', 'manifestation', 
            'negative evidence', 'probe', 'proof', 'quote', 'record', 'reference', 'research', 
            'result', 'sign', 'source', 'statistic', 'study', 'survey', 'testimonial', 
            'testimony', 'trial', 'verification', 'witness'
        ])
        strong_words = sum(1 for word in filtered_tokens if word in [
            'absolutely', 'always', 'can', 'certainly', 'compulsorily', 'could', 'definitely', 
            'have to', 'imperatively', 'inevitably', 'invariably', 'may', 'might', 'must', 
            'necessarily', 'need to', 'never', 'ought to', 'positively', 'shall', 'should', 
            'undoubtedly', 'unquestionably', 'will', 'would'
        ])
        reading_ease = flesch_reading_ease(text)
        
        # Speaker-level specific features
        num_segments = 1 if self.is_speaker_level else len(sent_tokenize(text))
        avg_words_per_segment = word_count / num_segments if num_segments > 0 else 0
        
        # Sentiment features
        blob = TextBlob(text)
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        
        # Rebuttal strength
        negation_count = sum(1 for token in tokens if token in [
            'against', 'barely', 'contradict', 'deny', 'disagree', 'except', 'hardly', 
            'neither', 'never', 'no', 'no one', 'nobody', 'none', 'nor', 'not', "n't", 
            'nothing', 'nowhere', 'oppose', 'rarely', 'refute', 'reject', 'scarcely', 
            'seldom', 'unless', 'without'
        ])
        rebuttal_sim = 0
        if opponent_text:
            opp_vec = self.fitted_vectorizer.transform([opponent_text]).toarray()
            text_vec = self.fitted_vectorizer.transform([text]).toarray()
            rebuttal_sim = cosine_similarity(text_vec, opp_vec)[0][0]
        
        # TF-IDF and embedding features
        tfidf_features = self.fitted_vectorizer.transform([text]).toarray()
        embedding = self.get_text_embeddings(text, max_features=50)
        
        features = np.hstack([
            tfidf_features,
            [[
                1 if is_for_position else 0,
                word_count,
                unique_words / word_count if word_count > 0 else 0,
                evidence_words,
                strong_words,
                reading_ease,
                len(text),
                num_segments,
                avg_words_per_segment,
                blob.sentiment.polarity,
                blob.sentiment.subjectivity,
                sentiment['compound'],
                rebuttal_sim,
                negation_count
            ]],
            [embedding]
        ])
        return features

    def predict_winner(self, new_argument, is_for_position=True, opponent_text=None):
        """Predict the winner of a debate argument."""
        if not all([self.model, self.fitted_vectorizer, self.fitted_scaler]):
            raise ValueError("Model, vectorizer, or scaler not trained/fitted yet")
        
        features = self.prepare_features(new_argument, is_for_position, opponent_text)
        features_scaled = self.fitted_scaler.transform(features)
        prediction = self.calibrated_model.predict(features_scaled)[0]
        probabilities = self.calibrated_model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': 'YES' if prediction == 1 else 'NO',
            'probability': probabilities[1],
            'confidence': max(probabilities)
        }

    def get_feature_importance(self):
        """Return feature importance for supported model types."""
        if self.model_type in ['random_forest', 'gradient_boosting', 'xgboost'] and self.model:
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        return None

    def save_model(self, directory='trained_model'):
        """Save the trained model and its components."""
        os.makedirs(directory, exist_ok=True)
        components = [
            (self.model, 'model.joblib'),
            (self.calibrated_model, 'calibrated_model.joblib'),
            (self.fitted_vectorizer, 'vectorizer.joblib'),
            (self.fitted_scaler, 'scaler.joblib'),
            (self.feature_names, 'feature_names.joblib')
        ]
        for component, filename in components:
            if component is not None:
                joblib.dump(component, os.path.join(directory, filename))
        print(f"Model and components saved to '{directory}' directory.")

    def load_model(self, directory='trained_model'):
        """Load a trained model and its components."""
        try:
            components = [
                ('model', 'model.joblib'),
                ('calibrated_model', 'calibrated_model.joblib'),
                ('fitted_vectorizer', 'vectorizer.joblib'),
                ('fitted_scaler', 'scaler.joblib'),
                ('feature_names', 'feature_names.joblib')
            ]
            for attr, filename in components:
                setattr(self, attr, joblib.load(os.path.join(directory, filename)))
            
            self.vectorizer = self.fitted_vectorizer
            self.scaler = self.fitted_scaler
            print(f"Model and components loaded from '{directory}' directory.")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new model instance...")
            self._initialize_default_model()

    def _initialize_default_model(self):
        """Initialize a default model if loading fails."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.calibrated_model = None
        self.fitted_vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        self.fitted_scaler = StandardScaler()
        
        dummy_texts = ["Sample text for initialization.", "Another sample text."]
        dummy_features = np.random.rand(2, 100)
        self.fitted_vectorizer.fit(dummy_texts)
        self.fitted_scaler.fit(dummy_features)
        self.vectorizer = self.fitted_vectorizer
        self.scaler = self.fitted_scaler

def load_and_prepare_data(file_path):
    """Load and preprocess debate data."""
    print("Loading Intelligence Squared debate data...")
    df = pd.read_csv(file_path).rename(columns={'text': 'argument_text'})
    
    initial_rows = len(df)
    df.dropna(subset=['argument_text'], inplace=True)
    if initial_rows - len(df) > 0:
        print(f"Dropped {initial_rows - len(df)} rows with missing 'argument_text'.")
    
    df['winner'] = df.apply(
        lambda row: 1 if row['speaker_name'] == row['conversation_winner'] and 
        row['conversation_winner'] != "It's a tie!" else 0, axis=1
    )
    
    print("\nDataset Overview:")
    print(f"- Total segments: {len(df)}")
    print(f"- Unique debates: {df['conversation_id'].nunique()}")
    print(f"- Unique speakers: {df['speaker_name'].nunique()}")
    print(f"- Winner distribution:\n{df['winner'].value_counts()}")
    print(f"- Position distribution:\n{df['speakertype'].value_counts()}")
    
    return df

def analyze_debate_dynamics(df):
    """Analyze debate patterns and speaker performance."""
    print("\n==================================================")
    print("DEBATE DYNAMICS ANALYSIS")
    print("==================================================")
    
    patterns = df.groupby(['speakertype', 'winner']).agg({
        'argument_text': [
            ('word_count', lambda x: x.str.split().str.len().mean()),
            ('std', lambda x: x.str.split().str.len().std()),
            ('sum', lambda x: x.str.split().str.len().sum())
        ],
        'argument_text': [('char_count', lambda x: x.str.len().mean())],
        'conversation_id': 'count'
    }).round(2)
    print("\nSpeaking patterns by position and outcome:")
    print(patterns)
    
    speaker_summary = df.groupby('speaker_name').agg({
        'winner': 'sum',
        'argument_text': [('word_count', lambda x: x.str.split().str.len().mean()), ('count', 'count')],
        'speakertype': 'first',
        'conversation_title': 'first'
    }).round(2)
    print("\nSpeaker Performance Summary:")
    print(speaker_summary)
    
    print("\nTopic-wise Results:")
    for topic in df['conversation_title'].unique():
        topic_df = df[df['conversation_title'] == topic]
        print(f"\n{topic}:")
        for _, row in topic_df.groupby('speaker_name').agg({
            'argument_text': lambda x: x.str.split().str.len().sum(),
            'winner': 'first',
            'speakertype': 'first'
        }).iterrows():
            print(f" {row['speakertype'].upper()}: {row.name} ({row['argument_text']} words) - "
                  f"{'WON' if row['winner'] == 1 else 'LOST'}")

def prepare_features(df, vectorizer, is_speaker_level=False):
    """Prepare features for training at segment or speaker level."""
    print(f"Extracting features for {'speaker' if is_speaker_level else 'segment'} level...")
    df['argument_text'] = df['argument_text'].astype(str).fillna('')
    
    if is_speaker_level:
        df = df.groupby(['conversation_id', 'speaker_name']).agg({
            'argument_text': lambda x: ' '.join(x.astype(str)),
            'winner': 'first',
            'speakertype': 'first',
            'conversation_title': 'first'
        }).reset_index()
    
    X_tfidf = vectorizer.fit_transform(df['argument_text']).toarray()
    features, embeddings_list = [], []
    sentiment_analyzer = SentimentIntensityAnalyzer()
    
    for _, row in df.iterrows():
        text = row['argument_text']
        tokens = word_tokenize(text.lower())
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
        
        word_count = len(filtered_tokens)
        unique_words = len(set(filtered_tokens))
        evidence_words = sum(1 for word in filtered_tokens if word in [
            'affidavit', 'analysis', 'anecdotal evidence', 'best evidence', 'case', 'citation', 
            'circumstantial evidence', 'clue', 'confirmation', 'corroboration', 'data', 'datum', 
            'demonstration', 'deposition', 'detail', 'direct evidence', 'documentation', 
            'empirical evidence', 'evidence', 'examination', 'example', 'exhibit', 'experiment', 
            'exploration', 'fact', 'figure', 'finding', 'illustration', 'indication', 
            'indirect evidence', 'information', 'inquiry', 'investigation', 'manifestation', 
            'negative evidence', 'probe', 'proof', 'quote', 'record', 'reference', 'research', 
            'result', 'sign', 'source', 'statistic', 'study', 'survey', 'testimonial', 
            'testimony', 'trial', 'verification', 'witness'
        ])
        strong_words = sum(1 for word in filtered_tokens if word in [
            'absolutely', 'always', 'can', 'certainly', 'compulsorily', 'could', 'definitely', 
            'have to', 'imperatively', 'inevitably', 'invariably', 'may', 'might', 'must', 
            'necessarily', 'need to', 'never', 'ought to', 'positively', 'shall', 'should', 
            'undoubtedly', 'unquestionably', 'will', 'would'
        ])
        reading_ease = flesch_reading_ease(text)
        
        blob = TextBlob(text)
        sentiment = sentiment_analyzer.polarity_scores(text)
        
        features.append([
            1 if row['speakertype'] == 'for' else 0,
            word_count,
            unique_words / word_count if word_count > 0 else 0,
            evidence_words,
            strong_words,
            reading_ease,
            len(text),
            blob.sentiment.polarity,
            blob.sentiment.subjectivity,
            sentiment['compound'],
            0,  # rebuttal_sim placeholder
            sum(1 for token in tokens if token in [
                'against', 'barely', 'contradict', 'deny', 'disagree', 'except', 'hardly', 
                'neither', 'never', 'no', 'no one', 'nobody', 'none', 'nor', 'not', "n't", 
                'nothing', 'nowhere', 'oppose', 'rarely', 'refute', 'reject', 'scarcely', 
                'seldom', 'unless', 'without'
            ])
        ])
        embeddings_list.append(DebateAnalyzer().get_text_embeddings(text, max_features=50))
    
    X = np.hstack([X_tfidf, np.array(features), np.array(embeddings_list)])
    feature_names = (
        [f'tfidf_{i}' for i in range(X_tfidf.shape[1])] +
        ['is_for_position', 'word_count', 'unique_word_ratio', 'evidence_words_count',
         'strong_words_count', 'reading_ease', 'segment_length', 'polarity', 
         'subjectivity', 'compound', 'rebuttal_sim', 'negation_count'] +
        [f'embedding_{i}' for i in range(50)]
    )
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Winner distribution:\n{df['winner'].value_counts()}")
    return X, df['winner'].values, feature_names, df

def add_argument(speaker_name, position, argument_text, conversation_history, conversation_title):
    """Add a new argument to the conversation history."""
    if not all([speaker_name, argument_text, conversation_title]):
        return conversation_history, "Please fill in all fields.", conversation_title, "", ""
    
    conversation_history = conversation_history or []
    conversation_history.append({'speaker': speaker_name, 'position': position, 'text': argument_text})
    return conversation_history, "", conversation_title, "", ""

def judge_debate(conversation_history, conversation_title):
    """Evaluate debate arguments and predict the winner."""
    if not conversation_history:
        return "No arguments provided.", ""
    
    test_debate = [{
        'conversation_id': uuid.uuid4().int & (1<<32)-1,
        'conversation_title': conversation_title,
        'conversation_winner': None,
        'speaker_name': entry['speaker'],
        'argument_text': entry['text'],
        'speakertype': entry['position'].lower()
    } for entry in conversation_history]
    
    test_df = pd.DataFrame(test_debate)
    speakers = test_df['speaker_name'].unique()
    history = [
        f"{row['speaker_name']}: \"{row['argument_text'][:97] + '...' if len(row['argument_text']) > 100 else row['argument_text']}\""
        for _, row in test_df.iterrows()
    ]
    
    speaker_texts = {
        speaker: ' '.join(test_df[test_df['speaker_name'] == speaker]['argument_text'].astype(str))
        for speaker in speakers
    }
    speaker_positions = {
        speaker: test_df[test_df['speaker_name'] == speaker]['speakertype'].iloc[0]
        for speaker in speakers
    }
    
    analyzer = DebateAnalyzer(model_type='random_forest', is_speaker_level=True)
    try:
        analyzer.load_model(directory='https://raw.githubusercontent.com/juctxy/debate/main/debate_2')
    except:
        print("Could not load model from URL. Using default model.")
        analyzer.load_model()
    
    results = {}
    for speaker, text in speaker_texts.items():
        is_for_position = (speaker_positions[speaker] == 'for')
        opponent_texts = ' '.join(txt for sp, txt in speaker_texts.items() if sp != speaker)
        try:
            result = analyzer.predict_winner(text, is_for_position, opponent_texts or None)
            results[speaker] = result['probability']
        except Exception as e:
            print(f"Error predicting for {speaker}: {e}")
            results[speaker] = np.random.random()
    
    predicted_winner = max(results, key=results.get)
    predicted_winner_prob = results[predicted_winner]
    
    history_html = f"""
    <div style='color: white; font-family: "Source Sans Pro", sans-serif;'>
        <h3 style='color: white; text-align: center;'>Debate: {conversation_title}</h3>
        <h4 style='color: white; text-align: center;'>History</h4>
        <ul style='color: white; list-style-type: none; padding-left: 0;'>
            {"".join(f"<li style='color: white; margin-bottom: 10px;'>{line}</li>" for line in history)}
        </ul>
    </div>
    """
    
    results_html = f"""
    <div style='color: white; font-family: "Source Sans Pro", sans-serif;'>
        <p style='color: white;'>Judgment: {predicted_winner} is predicted to WIN with {predicted_winner_prob:.3f} probability.</p>
    </div>
    """
    
    return history_html, results_html

# Gradio interface setup
CSS = """
.gradio-container {background-color: black !important; color: white !important;}
.gradio-container a {color: white !important;}
.gr-title {color: white !important; text-align: center !important; font-size: 26px !important; font-family: 'Source Sans Pro', sans-serif !important;}
.gr-row, .gr-row * {background-color: black !important; color: white !important; outline: none !important; box-shadow: none !important;}
.gr-radio, .gr-radio * {background-color: black !important; color: white !important;}
.gr-radio label {background-color: black !important; color: white !important;}
.gr-radio input[type="radio"] {background-color: black !important; color: white !important; border: 1px solid white !important;}
.gr-button {background-color: black !important; color: white !important; border: 1px solid white !important; cursor: pointer !important;}
.gr-button:hover {background-color: #222 !important;}
input:focus, textarea:focus, select:focus, button:focus {outline: none !important; box-shadow: none !important; border-color: white !important;}
.chat-container {max-width: 800px; margin: 0 auto; padding: 20px; color: white !important;}
.chat-message {margin: 10px 0; padding: 10px; border-radius: 5px; background-color: #333; color: white !important;}
"""

with gr.Blocks(css=CSS) as demo:
    conversation_history = gr.State([])
    gr.Markdown("""<div class="gr-title">Debate Arena: Argument Evaluator</div>""")
    
    conversation_title = gr.Textbox(lines=1, placeholder="Enter the debate topic...", 
                                  label="Debate Topic", elem_classes="gr-row")
    
    with gr.Row(elem_classes="gr-row"):
        speaker_name = gr.Textbox(lines=1, placeholder="Enter speaker name...", 
                                 label="Speaker Name", elem_classes="gr-row")
        position = gr.Radio(choices=["For", "Against"], label="Position", value="For", 
                           elem_classes="gr-row")
    
    argument_text = gr.Textbox(lines=3, placeholder="Enter your argument...", 
                              label="Argument", elem_classes="gr-row")
    
    add_button = gr.Button("Add Argument", elem_classes="gr-button")
    error_message = gr.Markdown("")
    finish_button = gr.Button("Finish and Judge", elem_classes="gr-button")
    history_output = gr.HTML()
    results_output = gr.HTML()
    
    add_button.click(
        fn=add_argument,
        inputs=[speaker_name, position, argument_text, conversation_history, conversation_title],
        outputs=[conversation_history, error_message, conversation_title, speaker_name, argument_text]
    ).then(
        fn=judge_debate,
        inputs=[conversation_history, conversation_title],
        outputs=[history_output, results_output]
    )
    
    finish_button.click(
        fn=judge_debate,
        inputs=[conversation_history, conversation_title],
        outputs=[history_output, results_output]
    )
    
    demo.load(
        fn=lambda title: (f"<div style='color: white;'><h3 style='color: white; text-align: center;'>"
                         f"Debate: {title or 'No topic entered'}</h3>"
                         f"<p style='color: white;'>No arguments yet. Add an argument to start the debate.</p></div>", ""),
        inputs=[conversation_title],
        outputs=[history_output, results_output]
    )

if __name__ == "__main__":
    demo.launch()
