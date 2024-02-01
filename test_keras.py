# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model
from losses import KerasLoss

# load model
model = load_model('ckpt-contextdesc/model-20000.hdf5',
	custom_objects={"KerasLoss": KerasLoss})
# summarize model.
model.summary()

# # load dataset
# dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # evaluate the model
# score = model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))