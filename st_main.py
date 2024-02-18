import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import numpy as np

# Define tokenizer:
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

# Load the model:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CamembertForSequenceClassification.from_pretrained("herelles/camembert-base-lupan")
model.to(device)

def prediction(segment_text):    
    test_ids = []
    test_attention_mask = []
    
    # Apply the tokenizer
    encoding = tokenizer(segment_text, padding="longest", return_tensors="pt")
    
    # Extract IDs and Attention Mask
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)
    
    # Forward pass, calculate logit predictions
    with torch.no_grad():
      output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))
    
    return np.argmax(output.logits.cpu().numpy()).flatten().item()

def main():
    st.header('Textual segments Hérelles prediction tool', divider='rainbow')
    
    segment_text = st.text_area(
    "Text to classify:",
    "Article 1 : Occupations ou utilisations du sol interdites\n\n"
    "1) Dans l’ensemble de la zone sont interdits :\n\n"
    "Les constructions destinées à l’habitation ne dépendant pas d’une exploitation agricole autres\n"
    "que celles visées à l’article 2 paragraphe 1).",
    height=170,
    )
        
    if st.button('Predict'):
        pred_id = prediction(segment_text)
        
        if pred_id == 0:
          pred_label = 'Not pertinent'
        elif pred_id == 1:
          pred_label = 'Pertinent (Soft)'
        elif pred_id == 2:
          pred_label = 'Pertinent (Strict, Non-verifiable)'
        elif pred_id == 3:
          pred_label = 'Pertinent (Strict, Verifiable)'
    
        st.write("Predicted Class: ", pred_label)
    
if __name__ == "__main__":
    main()
        
    
