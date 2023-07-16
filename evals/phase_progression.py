import numpy as np

import sklearn
from sklearn.linear_model import LinearRegression
from evals.datasets import DATASET_TO_NUM_CLASSES, IDX_TO_CLASS

# def regression_labels_for_class(labels, class_idx):
# 	# Assumes labels are ordered. Find the last occurrence of particular class.
# 	class_idx = IDX_TO_CLASS[class_idx]
# 	if len(np.argwhere(labels == class_idx)) > 0:
# 		transition_frame = np.argwhere(labels == class_idx)[-1, 0]
# 		return (np.arange(float(len(labels))) - transition_frame) / len(labels)
# 	else:
# 		return []

def regression_labels_for_class(labels, class_idx):
	# Assumes labels are ordered. Find the last occurrence of particular class.
	class_idx = IDX_TO_CLASS[class_idx]
	transition_frame = None
	if len(np.argwhere(labels == class_idx)) > 0:
		transition_frame = np.argwhere(labels == class_idx)[-1, 0]
	else:
		transition_frame = -1
	return (np.arange(float(len(labels))) - transition_frame) / len(labels)



def get_regression_labels(class_labels, num_classes):
	regression_labels = []
	for i in range(num_classes - 1):
		value=regression_labels_for_class(class_labels, i)
		if len(value) > 0:
			regression_labels.append(value)
	return np.stack(regression_labels, axis=1)

def get_targets_from_labels(all_class_labels, num_classes):
	all_regression_labels = []
	
	for class_labels in all_class_labels:
		all_regression_labels.append(get_regression_labels(class_labels, 
															num_classes))
	return all_regression_labels


def unnormalize(preds):
	seq_len = len(preds)
	return np.mean([i - pred * seq_len for i, pred in enumerate(preds)])


class VectorRegression(sklearn.base.BaseEstimator):
	"""Class to perform regression on multiple outputs."""

	def __init__(self, estimator):
		self.estimator = estimator

	def fit(self, x, y):
		_, m = y.shape
		# Fit a separate regressor for each column of y
		self.estimators_ = [sklearn.base.clone(self.estimator).fit(x, y[:, i])
							for i in range(m)]
		return self

	def predict(self, x):
		# Join regressors' predictions
		res = [est.predict(x)[:, np.newaxis] for est in self.estimators_]
		return np.hstack(res)

	def score(self, x, y):
		# Join regressors' scores
		res = [est.score(x, y[:, i]) for i, est in enumerate(self.estimators_)]
		return np.mean(res)


def fit_model(train_embs, train_labels, val_embs, val_labels):
	"""Linear Regression to regress to fraction completed."""

	train_embs = np.concatenate(train_embs, axis=0)
	train_labels = np.concatenate(train_labels, axis=0)
	val_embs = np.concatenate(val_embs, axis=0)
	val_labels = np.concatenate(val_labels, axis=0)

	lin_model = VectorRegression(LinearRegression())
	lin_model.fit(train_embs, train_labels)

	train_score = lin_model.score(train_embs, train_labels)
	val_score = lin_model.score(val_embs, val_labels)

	return train_score, val_score


def evaluate_phase_progression(train_data, val_data, action, ckpt_step, CONFIG, writer=None, verbose=False):

	train_embs = train_data['embs']
	val_embs = val_data['embs']
	num_classes = DATASET_TO_NUM_CLASSES[action]

	if not train_embs or not val_embs:
		raise Exception("All embeddings are NAN. Something is wrong with model.")

	val_labels = get_targets_from_labels(val_data['labels'], 
							num_classes)

	
	num_samples = len(train_data['embs'])

	train_scores = []
	val_scores = []
	for fraction_used in CONFIG.EVAL.CLASSIFICATION_FRACTIONS:
		num_samples_used = max(1, int(fraction_used * num_samples))
		train_embs = train_data['embs'][:num_samples_used]
		train_labels = get_targets_from_labels(
			train_data['labels'][:num_samples_used], num_classes)

		train_score, val_score = fit_model(train_embs, train_labels, val_embs, val_labels)

		if verbose:
			print('\n-----------------------------')
			print('Fraction: ', fraction_used)
			print('Train-Score: ', train_score)
			print('Val-Score: ', val_score)

		if writer:
			writer.add_scalar(f'phase_progression/train_{action}_{fraction_used}', train_score, global_step=ckpt_step)
			writer.add_scalar(f'phase_progression/val_{action}_{fraction_used}', val_score, global_step=ckpt_step)

		
		print(f'phase_progression/train_{action}_{fraction_used}', train_score, f"global_step={ckpt_step}")
		print(f'phase_progression/val_{action}_{fraction_used}', val_score, f"global_step={ckpt_step}")
		train_scores.append(train_score)
		val_scores.append(val_score)

	return train_scores, val_scores