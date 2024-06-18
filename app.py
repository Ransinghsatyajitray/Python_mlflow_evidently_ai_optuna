from src.irisclassification.pipelines.prediction_pipeline import CustomData, PredictPipeline

from flask import Flask, request, render_template, jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data=CustomData(
            
            SepalLengthCm = float(request.form.get('SepalLengthCm')),
            SepalWidthCm = float(request.form.get('SepalWidthCm')),
            PetalLengthCm = float(request.form.get('PetalLengthCm')),
            PetalWidthCm = float(request.form.get('PetalWidthCm'))
        )
        # this is my final data
        final_data = data.get_data_as_dataframe()
        
        predict_pipeline = PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result=pred                                                             # Some issue here XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=5000)