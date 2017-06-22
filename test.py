import pickle
import sys

class Test():

    def __init__(self, pathway):
        self.filepath = pathway

    def Unload(self):
        with open(self.filepath, 'rb') as f:
            data = (pickle.load(f))
            print(data, file = open('test.txt', 'w'))
            return data


    def print_lol(self, the_list, indent=False, level=0, output=sys.stdout):
        """This function takes one positional argument called "the_list", which is
            any Python list (of - possibly - nested lists). Each data item in the
            provided list is (recursively) printed to the screen on it's own line."""

        for each_item in the_list:
            if isinstance(each_item, list):
                print_lol(each_item, indent, level+1, output)
            else:
                if indent:
                    print("\t" * level, end='', file=output)
                print(each_item, file=output)


tester = Test("datasets/train.p")
data = tester.Unload()
tester.print_lol(data['features'][0][0][0])
