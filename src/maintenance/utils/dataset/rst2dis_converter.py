import os
import subprocess
from threading import Thread


def split_seq(seq: list, num_pieces: int):
    """ Just split a list of objects to num_pieces pieces """
    start = 0
    for i in range(num_pieces):
        stop = start + len(seq[i::num_pieces])
        yield seq[start:stop]
        start = stop

class RST2DISConverter(Thread):
    """ Use for multiprocess files *.rs3 -> *.dis conversion 
        with https://github.com/rst-workbench/rst-converter-service 
    """
    
    def __init__(self, base_url: str, batch: list, output_dir: str):
        Thread.__init__(self)
        self.base_url = base_url  # 'localhost:5000'
        self.batch = batch  # ['path/file1.rs3', 'path/file2.rs3', ...]
        self.output_dir = output_dir
        
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        
    def run(self):
        for file in self.batch:
            output_file = os.path.join(self.output_dir, file.split('/')[-1].replace('.rs3', '.dis'))
            cmd = f"curl -XPOST {self.base_url}/convert/rs3/dis -F input=@{file} > {output_file}"
            os.system(cmd)
