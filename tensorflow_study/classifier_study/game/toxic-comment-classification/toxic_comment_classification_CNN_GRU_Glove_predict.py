import os
import pickle
from keras.models import model_from_json

from toxic_comment_classification_CNN_GRU_Glove import DataClean, WordToken

# file path
DIR_ROOT = os.path.dirname(os.path.abspath("__file__"))
# mode saving path
DIR_MODEL = os.path.join(DIR_ROOT, 'model')
PREPROCESSOR_FILE = os.path.join(DIR_MODEL, 'preprocessor.pkl')
ARCHITECTURE_FILE = os.path.join(DIR_MODEL, 'cnn_gru_architecture.json')
WEIGHTS_FILE = os.path.join(DIR_MODEL, 'cnn_gru_weights.h5')


def load_model(architecture_file, weights_file):
    with open(architecture_file) as arch_json:
        architecture = arch_json.read()
    model = model_from_json(architecture)
    model.load_weights(weights_file)
    return model


def load_pipeline(preprocessor_file, architecture_file, weights_file):
    preprocessor = pickle.load(open(preprocessor_file, 'rb'))
    model = load_model(architecture_file, weights_file)
    return preprocessor, model


class PredictionPipeline(object):

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, text):
        self.preprocessor.__init__(text)
        features = self.preprocessor.transform_texts()
        pred = self.model.predict(features, batch_size=1024, verbose=1)
        return pred


if __name__ == "__main__":
    print("Loding model...")
    # *号表示：以元祖形式导入preprocess和model
    ppl = PredictionPipeline(*load_pipeline(PREPROCESSOR_FILE,
                                            ARCHITECTURE_FILE,
                                            WEIGHTS_FILE))
    print("Complete loding model!")

    # 被预测文本
    # sample_text = ['Corgi is stupid',
    #                'good boy',
    #                'School of AI is awesome',
    #                'FUCK']
    quit_flag = "start"
    while(quit_flag != "q"):
        # get input text and turn into array
        sample_text = [input("Please input English words: ")]
        # sorts
        class_names = ['toxic', 'severe_toxic', 'obscene',
                       'threat', 'insult', 'identity_hate']

        # 输出预测结果
        print("Prediction results: ")
        for text, toxicity in zip(sample_text, ppl.predict(sample_text)):
            print(f"{text}")
            for index, value in zip(class_names, toxicity):
                print(f"Toxicity of {index}:".ljust(27)+f"{value}")
        quit_flag = input("Type q for quit, Enter for go on: ")
