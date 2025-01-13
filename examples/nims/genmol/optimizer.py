
class MolecularOptimizer:

    def __init__(self, library = None, oracle = None, generator = None):
        self.library = library
        self.oracle = oracle
        self.generator = generator
        
    def run(self, iterations = 1, num_combined = 1, num_mutate = 1):
        
        for iter in range(iterations):
            self.library.export(num_combined)
            self.generator.produce(self.library.exported, num_mutate)
            self.library.update(self.oracle.evaluate(self.generator.molecules))
            
        return self