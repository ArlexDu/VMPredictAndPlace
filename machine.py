# coding=utf-8
class Machine:
    def __init__(self, memory, cpu):
        self.residueCPU = int(cpu)
        self.residueMemory = int(memory)
        self.vm_dict = {}

    def can_accommodate(self, memory_volume, cpu_volume):
        return (self.residueMemory >= int(memory_volume)) and (self.residueCPU >= int(cpu_volume))

    def assign_vm(self, name, memory_volume, cpu_volume, target):
        self.residueCPU -= int(cpu_volume)
        self.residueMemory -= int(memory_volume)
        if name in self.vm_dict:
            self.vm_dict[name] += 1
        else:
            self.vm_dict[name] = 1