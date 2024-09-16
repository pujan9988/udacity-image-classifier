from argparse import ArgumentParser

def arg_parse():
    parser = ArgumentParser(description="Image classifier")

    parser.add_argument("-num1",help="directory to save checkpoints",
                        type=int)
    parser.add_argument("-num2",help="No of epochs to train",
                        type=int)
    
    return parser.parse_args()
    

args = arg_parse()

num1 = args.num1 
num2 = args.num2 
print(type(num1))