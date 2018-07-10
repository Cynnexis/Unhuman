# -*- coding: utf-8 -*-

from enum import Enum


class Iris(Enum):
	"""
	Enumerate the type of Iris for the training dataset. Each enum has an integer value which match the tutorial
	"""
	SETOSA = 0
	VERSICOLOR = 1
	VIRGINICA = 2
	
	@staticmethod
	def to_str(iris) -> str:
		if isinstance(iris, Iris):
			if iris == Iris.SETOSA:
				return "Iris setosa"
			if iris == Iris.VERSICOLOR:
				return "Iris versicolor"
			if iris == Iris.VIRGINICA:
				return "Iris virginica"
		else:
			return "none"
	
	@staticmethod
	def to_int(iris) -> int:
		if isinstance(iris, Iris):
			return int(iris)
		else:
			return -1
	
	@staticmethod
	def from_int(value: int):
		if value == 0:
			return Iris.SETOSA
		elif value == 1:
			return Iris.VERSICOLOR
		elif value == 2:
			return Iris.VIRGINICA
		else:
			return None
