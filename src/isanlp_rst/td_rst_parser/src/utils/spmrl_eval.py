import os
import subprocess
import math
import re
class FScore(object):
	def __init__(self, recall, precision, fscore, complete_match, tagging_accuracy=100):
		self.recall = recall
		self.precision = precision
		self.fscore = fscore
		self.complete_match = complete_match
		self.tagging_accuracy = tagging_accuracy

	def __str__(self):
		if self.tagging_accuracy < 100:
			return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f}, CompleteMatch={:.2f}, TaggingAccuracy={:.2f})".format(
				self.recall, self.precision, self.fscore, self.complete_match, self.tagging_accuracy)
		else:
			return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f}, CompleteMatch={:.2f})".format(
				self.recall, self.precision, self.fscore, self.complete_match)


def evalb(evalb_dir, gold_path, predicted_path, output_path, ref_gold_path=None, label=True):
	assert os.path.exists(evalb_dir)
	evalb_program_path = os.path.join(evalb_dir, "evalb")
	evalb_spmrl_program_path = os.path.join(evalb_dir, "evalb_spmrl")
	assert os.path.exists(evalb_program_path) or os.path.exists(evalb_spmrl_program_path)
	#     output_path="output_check.txt"

	if os.path.exists(evalb_program_path):
		evalb_param_path = os.path.join(evalb_dir, "nk.prm")
	else:
		evalb_program_path = evalb_spmrl_program_path
		if label==True:
			evalb_param_path = os.path.join(evalb_dir, "spmrl.prm")
		else:
			evalb_param_path = os.path.join(evalb_dir, "unlabel_spmrl.prm")
	assert os.path.exists(evalb_program_path)
	assert os.path.exists(evalb_param_path)

	command = "{} -p {} {} {} > {}".format(
		evalb_program_path,
		evalb_param_path,
		gold_path,
		predicted_path,
		output_path,
	)
	subprocess.run(command, shell=True)

	fscore = FScore(math.nan, math.nan, math.nan, math.nan)
	with open(output_path) as infile:
		for line in infile:
			match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
			if match:
				fscore.recall = float(match.group(1))
			match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
			if match:
				fscore.precision = float(match.group(1))
			match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
			if match:
				fscore.fscore = float(match.group(1))
			match = re.match(r"Complete match\s+=\s+(\d+\.\d+)", line)
			if match:
				fscore.complete_match = float(match.group(1))
			match = re.match(r"Tagging accuracy\s+=\s+(\d+\.\d+)", line)
			if match:
				fscore.tagging_accuracy = float(match.group(1))
				break

	return fscore