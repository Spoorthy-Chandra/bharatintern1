data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)


count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


X_test_counts = count_vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
y_pred = model.predict(X_test_tfidf)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)

print('Cross-validation Scores:', cv_scores)
print('Mean CV Score:', cv_scores.mean())

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_tfidf)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
forest_model.fit(X_train_tfidf, y_train)


feature_importances = forest_model.feature_importances_

indices = np.argsort(feature_importances)[::-1]


plt.figure()
plt.title("Feature Importances")
plt.bar(range(X_train_tfidf.shape[1]), feature_importances[indices],
       color="b", align="center")
plt.xticks(range(X_train_tfidf.shape[1]), indices)
plt.xlim([-1, X_train_tfidf.shape[1]])
plt.show()



y_pred = model.predict(X_test_tfidf)

results = pd.DataFrame({'message': X_test, 'actual_label': y_test, 'predicted_label': y_pred})


spam_messages = results[results['predicted_label'] == 1]['message'].values


non_spam_messages = results[results['predicted_label'] == 0]['message'].values

print("Example Spam Messages:")
for msg in spam_messages[:5]:  
    print(msg)
    print("---")

print("\nExample Non-Spam Messages:")
for msg in non_spam_messages[:5]:  
    print(msg)
    print("---")

import pandas as pd


results = pd.DataFrame({
    'Message': X_test,  # SMS messages
    'Actual Label': y_test,  # Actual labels (0 for ham, 1 for spam)
    'Predicted Label': y_pred  # Predicted labels
})


results['Correct Classification'] = results['Actual Label'] == results['Predicted Label']


spam_messages = results[results['Predicted Label'] == 1]
non_spam_messages = results[results['Predicted Label'] == 0]


print("Sample of Spam Messages:")
print(spam_messages[['Message', 'Actual Label', 'Predicted Label']].head())

print("\nSample of Non-Spam Messages:")
print(non_spam_messages[['Message', 'Actual Label', 'Predicted Label']].head())

