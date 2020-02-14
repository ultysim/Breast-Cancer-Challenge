import pickle
from sklearn.ensemble import RandomForestClassifier
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('data')

model_dir = r'C:\Users\Simas\Desktop\Insight\Data Challenges\Breast-Cancer-Challenge\models\Trained_Forest'

class PredictCancer(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        print(args)
        user_query = args['data']
        print(user_query)

        data = user_query.split(',')
        data = [float(i) for i in data]

        loaded_model = pickle.load(open(model_dir, 'rb'))
        pred = loaded_model.predict([data])
        print(pred)

        # create JSON object
        if pred[0] == 1:
            output = {'prediction': 'Malignant'}
        else:
            output = {'prediction': 'Benign'}

        return output

api.add_resource(PredictCancer, '/')

if __name__ == '__main__':
    app.run(port=3000, debug=True)