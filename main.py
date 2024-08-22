import streamlit as st
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

# Charger un modèle GPT-2 plus grand, comme gpt2-large
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = TFGPT2LMHeadModel.from_pretrained("gpt2-large")

# Entrée de texte pour la génération
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors='tf')

# Créer l'attention_mask avec la même forme que input_ids
attention_mask = tf.ones_like(input_ids)

# Générer le texte avec une longueur maximale de 50 tokens
output = model.generate(
    input_ids, 
    attention_mask=attention_mask, 
    max_length=50, 
    num_return_sequences=1, 
    no_repeat_ngram_size=3,  # Empêche la répétition de groupes de 3 mots
    early_stopping=True,
    repetition_penalty=2.0,  # Pénalise la répétition excessive de mots
    do_sample=True,          # Activer l'échantillonnage
    temperature=0.7,         # Rend le texte plus varié
    top_k=50,                # Considère seulement les 50 tokens les plus probables à chaque étape
    top_p=0.95               # Nucleus sampling: prend en compte les tokens les plus probables jusqu'à ce que leur somme atteigne 95%
)

# Décoder et afficher le texte généré
generated_text = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
st.print(generated_text)
