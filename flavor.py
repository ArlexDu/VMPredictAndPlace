# coding=utf-8
class Flavor:
    """flavor according to input requirement."""

    def __init__(self, name, cpu, memory, date=None):
        self.name = name
        self.date = date
        self.cpu = int(cpu)
        self.memory = int(memory)
        self.predict_num = 0
        self.high = 0
