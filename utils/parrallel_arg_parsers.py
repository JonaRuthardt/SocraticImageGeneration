import argparse
import sys

class ParallelArgsParser:
    
    def __init__(self, **kwargs):
        self.__main_parser = argparse.ArgumentParser()
        self.__parallel_parsers = self.__main_parser.add_subparsers(**kwargs)
        self.__subs = dict()

    def add_parser(self,name:str,**kwargs) -> argparse.ArgumentParser:
        sp = self.__parallel_parsers.add_parser(name,**kwargs)
        self.__subs[name] = sp
        return sp
    
    def parse_args(self, args=None):
        if args is None:
            args = sys.argv
        subargs_dict = dict()
        rest = args
        for name, parser in self.__subs.items():
            subargs_dict[name], rest = parser.parse_known_args(rest)
        args_ns = argparse.Namespace(**subargs_dict)
        return args_ns
