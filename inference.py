from PIL import Image
from helpers import preprocess, postprocess
from cnn import load_model as cnn_model
from rnn import load_model as rnn_model


class Inference:
    def __init__(self):
        self.rnn = rnn_model()
        self.cnn = cnn_model()

    def infer(self, image):
        processed = preprocess(image)
        output_rnn = self.rnn(processed)
        output_cnn = self.cnn(processed)

        count_rnn = postprocess(output_rnn)
        count_cnn = postprocess(output_cnn)

        return {'cnn': count_cnn, 'rnn': count_rnn}


