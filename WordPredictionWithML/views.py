from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pickle
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('Notebooks/next_words.h5')
tokenizer = pickle.load(open('Notebooks/token.pkl', 'rb'))

@csrf_exempt
def Welcome(request):
    return render(request, 'index.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')
        if len(input_text.split()) < 3:
            return JsonResponse({'error': 'Minimum length for predicting next word is 3 words.'})
        
        prediction_result = predict_next_word(model, tokenizer, input_text)
        return JsonResponse({'prediction': prediction_result})
    else:
        return JsonResponse({'error': 'Invalid Request'})

def predict_next_word(model, tokenizer, text):
    words = text.split()
    words = words[-3:]
    sequence = tokenizer.texts_to_sequences([words])
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break

    return predicted_word
