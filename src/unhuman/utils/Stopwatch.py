# -*- coding: utf-8 -*-
import time


class Stopwatch:
	
	def __init__(self, start_now=False):
		self.__beginning = 0.
		self.__end = 0.
		self.__is_running = False
		
		if start_now:
			self.start()
	
	# STOPWATCH METHODS #
	
	def start(self):
		self.beginning = time.time()
	
	def stop(self) -> float:
		self.end = time.time()
		self.is_running = False
		return self.elapsed()
	
	def elapsed(self) -> float:
		if not self.is_running:
			return self.end - self.beginning
		else:
			return time.time() - self.beginning
	
	# GETTERS & SETTERS #
	
	def get_beginning(self) -> float:
		return self.__beginning
	
	def set_beginning(self, beginning: float) -> None:
		self.__beginning = beginning
	
	beginning = property(get_beginning, set_beginning)
	
	def get_end(self) -> float:
		return self.__end
	
	def set_end(self, end: float) -> None:
		self.__end = end
	
	end = property(get_end, set_end)
	
	def get_is_running(self) -> bool:
		return self.__is_running
	
	def set_is_running(self, is_running: bool) -> None:
		self.__is_running = is_running
	
	is_running = property(get_is_running, set_is_running)
	
	# OVERRIDE #
	
	def __eq__(self, o: object) -> bool:
		if isinstance(o, Stopwatch):
			return self.beginning == o.beginning and self.end == o.end and self.is_running == o.is_running
		else:
			return False
	
	def __str__(self) -> str:
		return self.__repr__()
	
	def __repr__(self) -> str:
		return "Stopwatch{beginning={:.2}s, end={:.2}s, is_running={}}".format(self.beginning, self.end, self.is_running)
