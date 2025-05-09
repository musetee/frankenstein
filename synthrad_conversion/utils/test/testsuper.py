class BaseDataLoader:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.create()
        self.test()

    def printbase(self):
        print("verify if I can use function only existed in superclass")

    def create(self):
        self.getc()
        
    def getc(self):
        self.c=0

    def test(self):
        # Base implementation that does something common for all subclasses
        print("test c in test:",self.c)
        print("Basic setup for all data loaders.")
        print("Basic values:", self.a, self.b)

class MonaiLoader(BaseDataLoader):
    def __init__(self, a, b):
        print("init monai")
        self.a = a
        self.b = b
        super().create()
        super().test()

        print("test c in monailoader:", self.c)
        super().printbase()

        super().getc()
        print("test c in super class:", self.c)

    def create(self):
        self.getc()
        # we want to use the value of superclass:
        #super().getc()

    def getc(self):
        self.c=1

        

if __name__=="__main__":
    monailoader=MonaiLoader(3, 4)
    #monailoader.test()
    print(monailoader.a, monailoader.b, monailoader.c)
    

