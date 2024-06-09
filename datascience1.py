{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c206c3ca",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:00.434387Z",
     "iopub.status.busy": "2024-06-09T09:00:00.434006Z",
     "iopub.status.idle": "2024-06-09T09:00:01.453385Z",
     "shell.execute_reply": "2024-06-09T09:00:01.451996Z"
    },
    "papermill": {
     "duration": 1.029103,
     "end_time": "2024-06-09T09:00:01.456300",
     "exception": false,
     "start_time": "2024-06-09T09:00:00.427197",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/sms-spam-collection-dataset/spam.csv\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ef980da9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:01.468738Z",
     "iopub.status.busy": "2024-06-09T09:00:01.468259Z",
     "iopub.status.idle": "2024-06-09T09:00:02.965240Z",
     "shell.execute_reply": "2024-06-09T09:00:02.964068Z"
    },
    "papermill": {
     "duration": 1.505775,
     "end_time": "2024-06-09T09:00:02.967862",
     "exception": false,
     "start_time": "2024-06-09T09:00:01.462087",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4977cc0c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:02.979801Z",
     "iopub.status.busy": "2024-06-09T09:00:02.978862Z",
     "iopub.status.idle": "2024-06-09T09:00:03.016561Z",
     "shell.execute_reply": "2024-06-09T09:00:03.015306Z"
    },
    "papermill": {
     "duration": 0.046612,
     "end_time": "2024-06-09T09:00:03.019413",
     "exception": false,
     "start_time": "2024-06-09T09:00:02.972801",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "56e02699",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:03.030968Z",
     "iopub.status.busy": "2024-06-09T09:00:03.030543Z",
     "iopub.status.idle": "2024-06-09T09:00:03.050765Z",
     "shell.execute_reply": "2024-06-09T09:00:03.049586Z"
    },
    "papermill": {
     "duration": 0.028709,
     "end_time": "2024-06-09T09:00:03.053069",
     "exception": false,
     "start_time": "2024-06-09T09:00:03.024360",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data = data[['v1', 'v2']]\n",
    "data.columns = ['label', 'message']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "809746eb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:03.064291Z",
     "iopub.status.busy": "2024-06-09T09:00:03.063889Z",
     "iopub.status.idle": "2024-06-09T09:00:03.076014Z",
     "shell.execute_reply": "2024-06-09T09:00:03.074812Z"
    },
    "papermill": {
     "duration": 0.020695,
     "end_time": "2024-06-09T09:00:03.078676",
     "exception": false,
     "start_time": "2024-06-09T09:00:03.057981",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "data['label'] = data['label'].map({'ham': 0, 'spam': 1})\n",
    "X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8b164c74",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:03.090016Z",
     "iopub.status.busy": "2024-06-09T09:00:03.089605Z",
     "iopub.status.idle": "2024-06-09T09:00:03.283556Z",
     "shell.execute_reply": "2024-06-09T09:00:03.282347Z"
    },
    "papermill": {
     "duration": 0.202652,
     "end_time": "2024-06-09T09:00:03.286181",
     "exception": false,
     "start_time": "2024-06-09T09:00:03.083529",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "count_vectorizer = CountVectorizer()\n",
    "X_train_counts = count_vectorizer.fit_transform(X_train)\n",
    "tfidf_transformer = TfidfTransformer()\n",
    "X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "379dc65c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:03.297730Z",
     "iopub.status.busy": "2024-06-09T09:00:03.297281Z",
     "iopub.status.idle": "2024-06-09T09:00:03.313658Z",
     "shell.execute_reply": "2024-06-09T09:00:03.312487Z"
    },
    "papermill": {
     "duration": 0.024827,
     "end_time": "2024-06-09T09:00:03.315940",
     "exception": false,
     "start_time": "2024-06-09T09:00:03.291113",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>MultinomialNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">MultinomialNB</label><div class=\"sk-toggleable__content\"><pre>MultinomialNB()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "MultinomialNB()"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = MultinomialNB()\n",
    "model.fit(X_train_tfidf, y_train)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "319a81ac",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:03.327731Z",
     "iopub.status.busy": "2024-06-09T09:00:03.327339Z",
     "iopub.status.idle": "2024-06-09T09:00:03.364825Z",
     "shell.execute_reply": "2024-06-09T09:00:03.363655Z"
    },
    "papermill": {
     "duration": 0.046342,
     "end_time": "2024-06-09T09:00:03.367410",
     "exception": false,
     "start_time": "2024-06-09T09:00:03.321068",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.9623318385650225\n",
      "Confusion Matrix:\n",
      " [[965   0]\n",
      " [ 42 108]]\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.96      1.00      0.98       965\n",
      "           1       1.00      0.72      0.84       150\n",
      "\n",
      "    accuracy                           0.96      1115\n",
      "   macro avg       0.98      0.86      0.91      1115\n",
      "weighted avg       0.96      0.96      0.96      1115\n",
      "\n"
     ]
    }
   ],
   "source": [
    "X_test_counts = count_vectorizer.transform(X_test)\n",
    "X_test_tfidf = tfidf_transformer.transform(X_test_counts)\n",
    "y_pred = model.predict(X_test_tfidf)\n",
    "print('Accuracy:', accuracy_score(y_test, y_pred))\n",
    "print('Confusion Matrix:\\n', confusion_matrix(y_test, y_pred))\n",
    "print('Classification Report:\\n', classification_report(y_test, y_pred))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f5417655",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:03.379674Z",
     "iopub.status.busy": "2024-06-09T09:00:03.379268Z",
     "iopub.status.idle": "2024-06-09T09:00:03.414385Z",
     "shell.execute_reply": "2024-06-09T09:00:03.413228Z"
    },
    "papermill": {
     "duration": 0.044208,
     "end_time": "2024-06-09T09:00:03.416869",
     "exception": false,
     "start_time": "2024-06-09T09:00:03.372661",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cross-validation Scores: [0.95627803 0.95403587 0.95510662 0.95398429 0.95847363]\n",
      "Mean CV Score: 0.9555756871152985\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)\n",
    "\n",
    "print('Cross-validation Scores:', cv_scores)\n",
    "print('Mean CV Score:', cv_scores.mean())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9f0d31f0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:03.429246Z",
     "iopub.status.busy": "2024-06-09T09:00:03.428814Z",
     "iopub.status.idle": "2024-06-09T09:00:03.687008Z",
     "shell.execute_reply": "2024-06-09T09:00:03.685851Z"
    },
    "papermill": {
     "duration": 0.26753,
     "end_time": "2024-06-09T09:00:03.689738",
     "exception": false,
     "start_time": "2024-06-09T09:00:03.422208",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.metrics import roc_curve, auc\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test_tfidf)[:, 1])\n",
    "roc_auc = auc(fpr, tpr)\n",
    "\n",
    "plt.figure()\n",
    "plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)\n",
    "plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')\n",
    "plt.xlim([0.0, 1.0])\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.xlabel('False Positive Rate')\n",
    "plt.ylabel('True Positive Rate')\n",
    "plt.title('Receiver Operating Characteristic (ROC)')\n",
    "plt.legend(loc=\"lower right\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "8ed9e675",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:00:03.703428Z",
     "iopub.status.busy": "2024-06-09T09:00:03.703028Z",
     "iopub.status.idle": "2024-06-09T09:01:08.174231Z",
     "shell.execute_reply": "2024-06-09T09:01:08.172999Z"
    },
    "papermill": {
     "duration": 64.486255,
     "end_time": "2024-06-09T09:01:08.182057",
     "exception": false,
     "start_time": "2024-06-09T09:00:03.695802",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "image/png": "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+QyXL1+uc+fO6Z577pEkpaWlydfXVy+//LLL2l999VXV1NRoxIgRbdpPQECA27eK+/j4tPhM1q5d67zHx1O33367YmJitHjx4hb7a95Pz549NWTIEP3xj3/U0aNHW8zxzScRv/1vExgYqBtuuEH19fXtWh/Q2XHGCLgE2Ww2LV++XI8++qhuv/12jRo1SldddZUOHTqkt956S3fccYeWLFkim83mfJT97NmzioyM1F//+lft37+/xZyJiYmSpN/97ncaNWqUunbtqp///OcKCAjQ448/rgULFujxxx9Xv379tHXrVv3jH/9o83qvv/56Pfvss8rNzdWBAweUkZGh7t27a//+/frP//xPTZgwQdOmTfvBPh9PNDQ0aOjQoRo5cqT27t2rZcuWadCgQbrvvvsknX9lQW5urubOnathw4bpvvvuc9b1799fjzzySJv2k5iYqOXLl+vZZ5/VDTfcoJ49e+quu+7Svffeq3nz5ik7O1sDBw7Uzp079dprr7mcnfKEt7e3li9frp///OdKSEhQdna2evXqpT179mj37t367//+b0nnb+wfNGiQ4uPjNX78eF133XWqqqpSaWmpPvvsM+d7lG655RYNGTJEiYmJCgkJ0Y4dO/TGG29o4sSJ7Vof0Ol10NNwwBXN3ePp7rzzzjtWenq6FRQUZPn7+1vXX3+99dhjj1k7duxw1nz22WfW/fffbwUHB1tBQUHWww8/bB05cqTF4+uWZVm///3vrcjISMvb29vl0f3Tp09b48aNs4KCgqzu3btbI0eOtI4dO9bq4/rHjx93u96//OUv1qBBg6yAgAArICDAiouLs5566ilr7969Hn8eY8eOtQICAlrUpqamWrfeemuL/muvvdblsfPmOd99911rwoQJVo8ePazAwEBr9OjR1okTJ1qMX7JkiRUXF2d17drVCg8Pt5588skWj8O3tm/LOv8qhREjRljdu3e3JDkf3T9z5oz19NNPW7169bK6detm3XHHHVZpaamVmprq8nh/8+P6a9eudZm3tdcpvPfee9bPfvYzq3v37lZAQIDVp08f65VXXnGp+eSTT6wxY8ZYdrvd6tq1qxUZGWnde++91htvvOGsefbZZ62kpCQrODjY6tatmxUXF2fNnz/f5RUHwJXEy7Iuwt2KAHCRrVy5UtnZ2dq+fXun/7MrAC4e7jECAAAwCEYA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++Kcy/FoL76+vkpMTFRxcbGzr6mpScXFxUpJSXE7JiUlxaVekoqKipz1MTExstvtLjW1tbUqKytz1qSkpOjUqVMqLy931mzZskVNTU1KTk725BAAAABaZ3lo9erVlp+fn7Vy5Urrww8/tCZMmGAFBwdbDofDsizLevTRR63p06c76//2t79ZXbp0sf75n//Z+uijj6y8vDyra9eu1s6dO501CxYssIKDg60333zT+uCDD6xf/OIXVkxMjPX//t//c9YMGzbMuu2226yysjLrvffes2JjY62srKw2r3vbtm2WJBqNRqPRaJ2wzZ8/39PI0i4eByPLsqxXXnnFuuaaayxfX18rKSnJ+vvf/+7clpqaao0dO9al/vXXX7duvPFGy9fX17r11lutt956y2V7U1OTNWvWLCs8PNzy8/Ozhg4dau3du9el5sSJE1ZWVpYVGBho2Ww2Kzs72/ryyy/bvObDhw93+D8qjUaj0Wi09rW3337b88DSDh6/xwgAAOByddk+lQYAAOApghEAAIBBMAIAADAIRgAAAAbBCAAAwPDob6V1NK/W/gw9AABAK6qqqlr8zdXWdKrH9YOCgpx/ZRcAAKAtPIk6nepSWhB/ph0AAHjosg1GJ06c6OglAACATubgwYNtru1UwaihoaGjlwAAADqZAwcOtLm2UwUjbr4GAACe+vTTT9tc26mCUXBwcEcvAQAAdDLvvvtum2s7VTBKTU3t6CUAAIBOJiMjo821nepx/R49eujUqVMdvQwAANCJXLZPpZ0+fbqjlwAAADqR8PBwj+o71RkjAACAH1OnOmMEAADwYyIYAQAAGAQjAAAAg2AEAABgEIwAAAAMghEAAIBBMAIAADAIRgAAAAbBCAAAwCAYAQAAGAQjAAAA4/8DWipEr9wPEbEAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "forest_model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "forest_model.fit(X_train_tfidf, y_train)\n",
    "\n",
    "\n",
    "feature_importances = forest_model.feature_importances_\n",
    "\n",
    "indices = np.argsort(feature_importances)[::-1]\n",
    "\n",
    "\n",
    "plt.figure()\n",
    "plt.title(\"Feature Importances\")\n",
    "plt.bar(range(X_train_tfidf.shape[1]), feature_importances[indices],\n",
    "       color=\"b\", align=\"center\")\n",
    "plt.xticks(range(X_train_tfidf.shape[1]), indices)\n",
    "plt.xlim([-1, X_train_tfidf.shape[1]])\n",
    "plt.show()\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "bc0c1c22",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:01:08.196158Z",
     "iopub.status.busy": "2024-06-09T09:01:08.195751Z",
     "iopub.status.idle": "2024-06-09T09:01:08.202929Z",
     "shell.execute_reply": "2024-06-09T09:01:08.201807Z"
    },
    "papermill": {
     "duration": 0.017121,
     "end_time": "2024-06-09T09:01:08.205427",
     "exception": false,
     "start_time": "2024-06-09T09:01:08.188306",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "y_pred = model.predict(X_test_tfidf)\n",
    "\n",
    "results = pd.DataFrame({'message': X_test, 'actual_label': y_test, 'predicted_label': y_pred})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "9cc625bf",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:01:08.220570Z",
     "iopub.status.busy": "2024-06-09T09:01:08.220184Z",
     "iopub.status.idle": "2024-06-09T09:01:08.230680Z",
     "shell.execute_reply": "2024-06-09T09:01:08.229431Z"
    },
    "papermill": {
     "duration": 0.021046,
     "end_time": "2024-06-09T09:01:08.233187",
     "exception": false,
     "start_time": "2024-06-09T09:01:08.212141",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "spam_messages = results[results['predicted_label'] == 1]['message'].values\n",
    "\n",
    "\n",
    "non_spam_messages = results[results['predicted_label'] == 0]['message'].values\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "24417bfb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:01:08.248487Z",
     "iopub.status.busy": "2024-06-09T09:01:08.248084Z",
     "iopub.status.idle": "2024-06-09T09:01:08.255206Z",
     "shell.execute_reply": "2024-06-09T09:01:08.254096Z"
    },
    "papermill": {
     "duration": 0.018048,
     "end_time": "2024-06-09T09:01:08.257978",
     "exception": false,
     "start_time": "2024-06-09T09:01:08.239930",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Example Spam Messages:\n",
      "Congratulations ur awarded either å£500 of CD gift vouchers & Free entry 2 our å£100 weekly draw txt MUSIC to 87066 TnCs www.Ldew.com1win150ppmx3age16\n",
      "---\n",
      "Free tones Hope you enjoyed your new content. text stop to 61610 to unsubscribe. help:08712400602450p Provided by tones2you.co.uk\n",
      "---\n",
      "URGENT! Your mobile was awarded a å£1,500 Bonus Caller Prize on 27/6/03. Our final attempt 2 contact U! Call 08714714011\n",
      "---\n",
      "This is the 2nd time we have tried 2 contact u. U have won the å£750 Pound prize. 2 claim is easy, call 087187272008 NOW1! Only 10p per minute. BT-national-rate.\n",
      "---\n",
      "We tried to call you re your reply to our sms for a video mobile 750 mins UNLIMITED TEXT free camcorder Reply or call now 08000930705 Del Thurs\n",
      "---\n",
      "\n",
      "Example Non-Spam Messages:\n",
      "Funny fact Nobody teaches volcanoes 2 erupt, tsunamis 2 arise, hurricanes 2 sway aroundn no 1 teaches hw 2 choose a wife Natural disasters just happens\n",
      "---\n",
      "I sent my scores to sophas and i had to do secondary application for a few schools. I think if you are thinking of applying, do a research on cost also. Contact joke ogunrinde, her school is one me the less expensive ones\n",
      "---\n",
      "We know someone who you know that fancies you. Call 09058097218 to find out who. POBox 6, LS15HB 150p\n",
      "---\n",
      "Only if you promise your getting out as SOON as you can. And you'll text me in the morning to let me know you made it in ok.\n",
      "---\n",
      "I'll text carlos and let you know, hang on\n",
      "---\n"
     ]
    }
   ],
   "source": [
    "print(\"Example Spam Messages:\")\n",
    "for msg in spam_messages[:5]:  \n",
    "    print(msg)\n",
    "    print(\"---\")\n",
    "\n",
    "print(\"\\nExample Non-Spam Messages:\")\n",
    "for msg in non_spam_messages[:5]:  \n",
    "    print(msg)\n",
    "    print(\"---\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "979591d6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-09T09:01:08.272818Z",
     "iopub.status.busy": "2024-06-09T09:01:08.272370Z",
     "iopub.status.idle": "2024-06-09T09:01:08.296053Z",
     "shell.execute_reply": "2024-06-09T09:01:08.294418Z"
    },
    "papermill": {
     "duration": 0.034994,
     "end_time": "2024-06-09T09:01:08.299410",
     "exception": false,
     "start_time": "2024-06-09T09:01:08.264416",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sample of Spam Messages:\n",
      "                                                Message  Actual Label  \\\n",
      "812   Congratulations ur awarded either å£500 of CD ...             1   \n",
      "1992  Free tones Hope you enjoyed your new content. ...             1   \n",
      "2952  URGENT! Your mobile was awarded a å£1,500 Bonu...             1   \n",
      "5567  This is the 2nd time we have tried 2 contact u...             1   \n",
      "3997  We tried to call you re your reply to our sms ...             1   \n",
      "\n",
      "      Predicted Label  \n",
      "812                 1  \n",
      "1992                1  \n",
      "2952                1  \n",
      "5567                1  \n",
      "3997                1  \n",
      "\n",
      "Sample of Non-Spam Messages:\n",
      "                                                Message  Actual Label  \\\n",
      "3245  Funny fact Nobody teaches volcanoes 2 erupt, t...             0   \n",
      "944   I sent my scores to sophas and i had to do sec...             0   \n",
      "1044  We know someone who you know that fancies you....             1   \n",
      "2484  Only if you promise your getting out as SOON a...             0   \n",
      "2973         I'll text carlos and let you know, hang on             0   \n",
      "\n",
      "      Predicted Label  \n",
      "3245                0  \n",
      "944                 0  \n",
      "1044                0  \n",
      "2484                0  \n",
      "2973                0  \n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "\n",
    "results = pd.DataFrame({\n",
    "    'Message': X_test,  # SMS messages\n",
    "    'Actual Label': y_test,  # Actual labels (0 for ham, 1 for spam)\n",
    "    'Predicted Label': y_pred  # Predicted labels\n",
    "})\n",
    "\n",
    "\n",
    "results['Correct Classification'] = results['Actual Label'] == results['Predicted Label']\n",
    "\n",
    "\n",
    "spam_messages = results[results['Predicted Label'] == 1]\n",
    "non_spam_messages = results[results['Predicted Label'] == 0]\n",
    "\n",
    "\n",
    "print(\"Sample of Spam Messages:\")\n",
    "print(spam_messages[['Message', 'Actual Label', 'Predicted Label']].head())\n",
    "\n",
    "print(\"\\nSample of Non-Spam Messages:\")\n",
    "print(non_spam_messages[['Message', 'Actual Label', 'Predicted Label']].head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e405a85",
   "metadata": {
    "papermill": {
     "duration": 0.006599,
     "end_time": "2024-06-09T09:01:08.313305",
     "exception": false,
     "start_time": "2024-06-09T09:01:08.306706",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "607de343",
   "metadata": {
    "papermill": {
     "duration": 0.006434,
     "end_time": "2024-06-09T09:01:08.326413",
     "exception": false,
     "start_time": "2024-06-09T09:01:08.319979",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0051978",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-06-04T19:43:23.016727Z",
     "iopub.status.busy": "2024-06-04T19:43:23.016350Z",
     "iopub.status.idle": "2024-06-04T19:43:23.036943Z",
     "shell.execute_reply": "2024-06-04T19:43:23.035881Z",
     "shell.execute_reply.started": "2024-06-04T19:43:23.016698Z"
    },
    "papermill": {
     "duration": 0.006381,
     "end_time": "2024-06-09T09:01:08.339425",
     "exception": false,
     "start_time": "2024-06-09T09:01:08.333044",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 483,
     "sourceId": 982,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30715,
   "isGpuEnabled": false,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 72.523114,
   "end_time": "2024-06-09T09:01:09.973355",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-06-09T08:59:57.450241",
   "version": "2.5.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
