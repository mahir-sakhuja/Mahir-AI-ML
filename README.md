# Mahir AI-ML Repository

This repository contains the full set of interactive projects from the **AI/ML course** (Day 1 → Day 7). Each sub-folder is a standalone Streamlit application that can be run locally.

## How to run a project locally
```bash
# 1️⃣ Clone the repo
git clone https://github.com/mahirsakhuja/Mahir-AI-ML.git
cd Mahir-AI-ML

# 2️⃣ Navigate to the desired day/project folder, for example:
cd "Day 1 Course Material/Project 1 Data_visualisation"

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Streamlit app
streamlit run app.py
```

### Important notes
* **Gemini API Key:** For the Day 7 RAG apps, you will be prompted to paste your free Gemini API key in the Streamlit sidebar. You can obtain a free key at [Google AI Studio](https://aistudio.google.com/).
* **Local Models:** Machine Learning models (like the KNN and CNN classifiers) are trained locally, and their serialization files (like `.pkl` and `.h5`/`.keras` files) are saved inside their respective directories for immediate inference.

---

## Complete Project Directory

| Day | Project Name | Description | Path |
| :--- | :--- | :--- | :--- |
| **Day 1** | **Google Play Store Analytics** | Interactive data cleaning, stats, seaborn & plotly charts on playstore dataset. | [`Day 1 Course Material/Project 1 Data_visualisation`](https://mahir-ai-ml-zuxln6v7ibrnxiqdxtmgaf.streamlit.app/) |
| **Day 2** | **Canada Per Capita Income** | Linear Regression model predicting income trends up to the year 2050. | [`Day 2 Course Material/Project 3 canada_per_capita_income`](https://mahir-ai-ml-d8bun3jxtwecsifl9zxappk.streamlit.app/) |
| **Day 3** | **Linear Regression (House Prices)** | Linear Regression model deployment for predicting housing costs. | [`Day 3 Course Material/Deployment Linear Regression`](https://mahir-ai-ml-azwb38tuwdmbhk2jxpe66f.streamlit.app/) |
| **Day 3** | **HR Employee Retention** | Logistic regression classifier determining probability of employee departure. | [`Day 3 Course Material/Project 4 Logistic Regression`](file:///c:/Users/Mahir%20sakhuja/New%20folder%20(4)/Day%203%20Course%20Material/Project%204%20Logistic%20Regression) |
| **Day 4** | **Gender Image Classification (ML)** | Logistic Regression model trained on raw flattened image vectors. | [`Day 4 Course Material/Project 5 Male and Female image classification`](file:///c:/Users/Mahir%20sakhuja/New%20folder%20(4)/Day%204%20Course%20Material/Project%205%20Male%20and%20Female%20image%20classification) |
| **Day 5** | **Gender Image Classification (CNN)** | Deep Convolutional Neural Network (CNN) in Keras classifying images. | [`Day 5 Course Material/Project 7 Male and Female image classification using CNN`](file:///c:/Users/Mahir%20sakhuja/New%20folder%20(4)/Day%205%20Course%20Material/Project%207%20Male%20and%20Female%20image%20classification%20using%20CNN) |
| **Day 6** | **Logical AND Gate Neural Net** | Single neuron neural network solving standard logical AND operations. | [`Day 6 Course Material/AND_using_Keras_Neural_Netwrok`](file:///c:/Users/Mahir%20sakhuja/New%20folder%20(4)/Day%206%20Course%20Material/AND_using_Keras_Neural_Netwrok) |
| **Day 6** | **Logical XOR Gate Neural Net** | Multi-layer Perceptron (MLP) neural network solving non-linear XOR operations. | [`Day 6 Course Material/XOR_using_Neural_Netwrok`](file:///c:/Users/Mahir%20sakhuja/New%20folder%20(4)/Day%206%20Course%20Material/XOR_using_Neural_Netwrok) |
| **Day 6** | **Iris Species KNN Deployment** | K-Nearest Neighbors model classifying iris flower species. | [`Day 6 Course Material/Project 08 KNN Deployement`](file:///c:/Users/Mahir%20sakhuja/New%20folder%20(4)/Day%206%20Course%20Material/Project%2008%20KNN%20Deployement) |
| **Day 6** | **Eye Gender Detection (CNN)** | Deep CNN identifying gender from cropped images of eyes. | [`Day 6 Course Material/Project 09 Detection of Male and Female Eyes from image Datsete`](file:///c:/Users/Mahir%20sakhuja/New%20folder%20(4)/Day%206%20Course%20Material/Project%2009%20Detection%20of%20Male%20and%20Female%20Eyes%20from%20image%20Datsete) |
| **Day 7** | **Samsung Washing Machine RAG** | Conversational agent using manual embeddings to answer product questions. | [`Day 07 Course Material/Samsung_RAG Deployment on Streamlit`](file:///c:/Users/Mahir%20sakhuja/New%20folder%20(4)/Day%2007%20Course%20Material/Samsung_RAG%20Deployment%20on%20Streamlit) |
| **Day 7** | **Star Health Insurance RAG** | Technical RAG Chatbot detailing health insurance options and coverage details. | [`Day 07 Course Material/Project 10 Star Health Insurance Policies Building RAG Chatbots for Technical Documentation`](file:///c:/Users/Mahir%20sakhuja/New%20folder%20(4)/Day%2007%20Course%20Material/Project%2010%20Star%20Health%20Insurance%20Policies%20Building%20RAG%20Chatbots%20for%20Technical%20Documentation) |

---

Feel free to browse each project folder, view the source code, check the individual `requirements.txt` lists, and launch the Streamlit dashboards!
