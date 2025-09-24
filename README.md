# English–French Neural Machine Translation (Seq2Seq with Attention)

This project implements a **Neural Machine Translation (NMT)** model to translate English sentences into French using a **Sequence-to-Sequence (Seq2Seq) model with Attention** in TensorFlow/Keras.  
It uses the **eng-fra.txt** dataset from [ManyThings](http://www.manythings.org/anki/).

---

## 📂 Project Structure

Language Translation/
│── eng-fra.txt # Dataset (English–French sentence pairs, tab separated)
│── main.py # Main training + inference script
│── eng_fra_translator.h5 # Saved trained model (after running main.py)
│── eng_tokenizer.pkl # Pickled English tokenizer
│── fra_tokenizer.pkl # Pickled French tokenizer
│── README.md # Project documentation

🧠 Model Architecture

Encoder: Embedding + LSTM (returns sequences + states)

Decoder: Embedding + LSTM (initialized with encoder states)

Attention Layer: Focuses on relevant encoder outputs at each step

Dense Layer: Predicts next word (softmax over French vocabulary)

📊 Dataset Info

Format: English \t French

Example:

Go.    Va !
Hi.    Salut !
Run!   Cours !


Script automatically preprocesses and adds <start> and <end> tokens to French sentences.

⚡ Tips

Training with the full dataset can be slow on CPU.
Reduce training size with max_lines in main.py (e.g., max_lines=2000).

You can adjust hyperparameters:

embedding_dim = 256

units = 512

epochs = 10

📌 Requirements

Python 3.8+

TensorFlow 2.x

NumPy

✨ Future Improvements

Add Beam Search for better translation quality.

Train with a larger dataset for improved fluency.

Build a Streamlit/Gradio web app for interactive translations.

👨‍💻 Author

Shreyash Yenkar
📧 shreyash.y14@gmail.com

🔗 LinkedIn
