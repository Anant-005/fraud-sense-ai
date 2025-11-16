import streamlit as st
import tensorflow as tf
import numpy as np

# --- 1. Load Your "Fraud Sense AI" Model ---
@st.cache_resource
def load_my_model():
    print("Loading Fraud Sense AI model...")
    model = tf.keras.models.load_model('fraud_sense_model.keras')
    print("Model loaded successfully.")
    return model

model = load_my_model()

# --- 2. Define Your Prediction Threshold ---
FINAL_THRESHOLD = 0.1

# --- 3. Build the Streamlit Web Interface ---

st.title("Fraud Sense AI 🤖")
st.subheader("Welcome to Fraud Sense AI: A Hybrid LLM+CNN Model for SMS Fraud Detection")
st.text("Enter an SMS message below to see if it's fraudulent.")

user_input = st.text_area("Enter message here:", "Click here to win a free prize...")

if st.button("Analyze Message"):
    if user_input:
        # --- 4. Make a Prediction ---
        
        # Create the NumPy array (this was correct)
        input_data = np.array([user_input])
        
        # --- THIS IS THE FINAL FIX ---
        # We must convert the input array to a tf.data.Dataset
        # just like we did in Colab.
        pred_dataset = tf.data.Dataset.from_tensor_slices(input_data)
        pred_dataset = pred_dataset.batch(1) # Batch it with batch size 1
        
        # Make the prediction on the dataset
        prediction_prob = model.predict(pred_dataset)
        
        # Get the single probability score
        prob_score = prediction_prob[0][0]
        
        # --- 5. Display the Result ---
        
        if prob_score > FINAL_THRESHOLD:
            st.error(f"🚨 ALERT: This looks like FRAUD (Spam)!", icon="🚨")
            st.write(f"**Confidence Score:** {prob_score*100:.2f}%")
        else:
            st.success(f"✅ SAFE: This looks like a normal message (Ham).", icon="✅")
            st.write(f"**Fraud Score:** {prob_score*100:.2f}%")
            
    else:
        st.warning("Please enter a message to analyze.")

st.caption("Developed by Anant Singh")