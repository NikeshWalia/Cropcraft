from flask import Flask, render_template, request, Markup
import pandas as pd
from utils.fertilizer import fertilizer_dict
import os
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import pickle

# Load models
classifier = load_model(r'E:\\extra\\crop\\Trained_model.h5')

# Dummy evaluation to build metrics and suppress the warning
dummy_data = np.zeros((1, 64, 64, 3))
dummy_label = np.zeros((1,))
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
classifier.evaluate(dummy_data, dummy_label)

crop_recommendation_model_path = ('E:\\extra\\crop\\Crop_Recommendation.pkl')
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

app = Flask(__name__)

@app.route('/fertilizer-predict', methods=['POST'])
def fertilizer_recommend():
    try:
        crop_name = str(request.form['cropname'])
        N_filled = int(request.form['nitrogen'])
        P_filled = int(request.form['phosphorous'])
        K_filled = int(request.form['potassium'])

        df = pd.read_csv(r'E:\\extra\\crop\\Data\\Crop_NPK.csv')

        N_desired = df[df['Crop'] == crop_name]['N'].iloc[0]
        P_desired = df[df['Crop'] == crop_name]['P'].iloc[0]
        K_desired = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = N_desired - N_filled
        p = P_desired - P_filled
        k = K_desired - K_filled

        key1 = "NNo" if n == 0 else ("NHigh" if n < 0 else "Nlow")
        key2 = "PNo" if p == 0 else ("PHigh" if p < 0 else "Plow")
        key3 = "KNo" if k == 0 else ("KHigh" if k < 0 else "Klow")

        abs_n = abs(n)
        abs_p = abs(p)
        abs_k = abs(k)

        response1 = Markup(str(fertilizer_dict[key1]))
        response2 = Markup(str(fertilizer_dict[key2]))
        response3 = Markup(str(fertilizer_dict[key3]))

        return render_template('Fertilizer-Result.html', recommendation1=response1,
                               recommendation2=response2, recommendation3=response3,
                               diff_n=abs_n, diff_p=abs_p, diff_k=abs_k)
    except Exception as e:
        return str(e)

def pred_pest(pest):
    try:
        test_image = image.load_img(pest, target_size=(64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = np.argmax(classifier.predict(test_image), axis=-1)
        return result
    except Exception as e:
        return 'x'

@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")

@app.route("/CropRecommendation.html")
def crop():
    return render_template("CropRecommendation.html")

@app.route("/FertilizerRecommendation.html")
def fertilizer():
    return render_template("FertilizerRecommendation.html")

@app.route("/PesticideRecommendation.html")
def pesticide():
    return render_template("PesticideRecommendation.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        pred = pred_pest(pest=file_path)
        if pred == 'x':
            return render_template('unaptfile.html')

        pest_identified = ['aphids', 'armyworm', 'beetle', 'bollworm', 'earthworm', 
                           'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem borer'][pred[0]]

        return render_template(pest_identified + ".html", pred=pest_identified)

@app.route('/crop_prediction', methods=['POST'])
def crop_prediction():
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['potassium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        return render_template('crop-result.html', prediction=final_prediction, pred='img/crop/' + final_prediction + '.jpg')

if __name__ == '__main__':
    app.run(debug=True)
