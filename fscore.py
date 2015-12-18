class fscore(object):
	def __init__(self):
		self.true_pos = 0.0
		self.true_neg = 0.0
		self.false_pos = 0.0
		self.false_neg = 0.0

	def precision(self):
		return (self.true_pos)/(self.true_pos + self.false_pos)

	def recall(self):
		return (self.true_pos)/(self.true_pos + self.false_neg)

	def calculate_f_score(self):
		upper_bound = (2.0*self.precision()*self.recall())
		lower_bound = (self.precision()+self.recall())
		return upper_bound/lower_bound