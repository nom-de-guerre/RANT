import ctypes as ct
import time as time

NNmL = ct.cdll.LoadLibrary('libRANT.so')

NNmL.AllocNNm.restype = ct.POINTER (ct.c_int)

NNmL.BuildTrainingSet.restype = ct.POINTER (ct.c_int)

NNmL.LoadMNIST.restype = ct.POINTER (ct.c_int)
NNmL.LoadMNIST.argtypes = [ct.c_char_p]

NNmL.LoadCSVClass.restype = ct.POINTER (ct.c_int)
NNmL.LoadCSVClass.argtypes = [ct.c_char_p, ct.c_int, ct.POINTER (ct.c_int)]

NNmL.LoadCSV.restype = ct.POINTER (ct.c_int)
NNmL.LoadCSV.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int, ct.POINTER (ct.c_int)]

NNmL.SetUseSGD.argtypes = [ct.POINTER (ct.c_int), ct.c_double]

NNmL.Inference.restype = ct.c_double
NNmL.Inference.argtypes = [ct.POINTER (ct.c_int), ct.POINTER (ct.c_int)]

NNmL.Classify.argtypes = [ct.POINTER (ct.c_int), ct.POINTER (ct.c_int)]

NNmL.ClassifyVec.argtypes = [ct.POINTER (ct.c_int), ct.POINTER (ct.c_int)]
NNmL.ClassifyVec.restype = ct.POINTER (ct.c_double)

NNmL.SetStopLoss.argtypes = [ct.POINTER (ct.c_int), ct.c_double]

NNmL.Loss.restype = ct.c_double

NNmL.ExtractTuple.restype = ct.POINTER (ct.c_int)
NNmL.ExtractTuple.argtypes = [ct.POINTER (ct.c_int), ct.c_int]

NNmL.Answer.restype = ct.c_double
NNmL.AnswerVec.restype = ct.POINTER (ct.c_double)

class NNm:

	def __init__ (self, hidden, Nin, Nout, seed = None):

		self.Nin = Nin
		self.Nout = Nout

		if (seed is None):
			seed = int (time.time ())

		NNmL.SeedRandom (seed)
		self.Model = NNmL.AllocNNm (hidden, Nin, Nout)

	def SetKeepAlive (self, N):
		NNmL.SetKeepAlive (self.Model, N)

	def SetStopLoss (self, haltVal):
		NNmL.SetStopLoss (self.Model, haltVal)

	def SetSGD (self, percentBatchSize):
		NNmL.SetUseSGD (self.Model, percentBatchSize)

	def SetMaxIterations (self, N):
		NNmL.SetUseSGD (self.Model, N)

	def AddPreProcessing (self, dataset):
		NNmL.SetPreProcessing (self.Model, dataset)

	def AddDense (self, N, ADAM = True):
		NNmL.AddDense (self.Model, N, ADAM)

	def AddConvFilter (self, N, width, stride, ADAM = True):
		NNmL.AddFilter (self.Model, N, width, stride, ADAM)

	def AddConvMaxPool (self, N, width, stride):
		NNmL.AddMaxPool (self.Model, N, width, stride)

	def AddMSE (self, ADAM = True):
		NNmL.AddMSE (self.Model, ADAM)

	def AddSoftmax (self, ADAM = True):
		NNmL.AddSoftmax (self.Model, ADAM)

	def AddMultiCLayer (self, ADAM = True):
		NNmL.AddMultiCLayer (self.Model, ADAM)

	def TestSet (self):
		return NNmL.BuildTrainingSet (32)

	def Train (self, TrainingSet, steps):
		NNmL.Train (self.Model, TrainingSet, steps)

	def Inference (self, predictors):
		return NNmL.Inference (self.Model, predictors)

	def Classify (self, predictors):
		return int (NNmL.Classify (self.Model, predictors))

	def ClassifyVec (self, predictors):
		__y = NNmL.ClassifyVec (self.Model, predictors)
		y = []
		for i in range (self.Nout):
			y.append (__y[i])

		return y

	def Arch (self):
		NNmL.Arch (self.Model)

	def Shape (self):
		NNmL.Shape (self.Model)

	def Loss (self):
		return NNmL.Loss (self.Model)

	def Steps (self):
		return NNmL.Steps (self.Model)

	def LoadMNIST (self, path):
		c_str = path.encode ("utf-8")
		return NNmL.LoadMNIST (c_str)

	def LoadFileClass (self, fileName, accept):
		Clist = (ct.c_int * len (accept)) (*accept)
		columns = len (accept)
		c_str = fileName.encode ("utf-8")
		return NNmL.LoadCSVClass (c_str, columns, Clist);

	def LoadFile (self, fileName, Nin, Nout, accept):
		Clist = (ct.c_int * len (accept)) (*accept)
		c_str = fileName.encode ("utf-8")
		return NNmL.LoadCSV (c_str, Nin + Nout, Nin, Nout, Clist);

	def DisplayData (self, O):
		return NNmL.DisplayData (O)

	def ExtractTuple (self, O, index):
		return NNmL.ExtractTuple (O, index)

	def Answer (self, O, index):
		y = NNmL.Answer (O, index)
#		y_list = []
#		y_list.append (y[0])

#		return y_list
		return y

	def AnswerVec (self, O, index):
		__y = NNmL.AnswerVec (O, index)
		y = []
		for i in range (self.Nout):
			y.append (__y[i])

		return y

	def DataLen (self, O):
		return NNmL.DataLen (O)

