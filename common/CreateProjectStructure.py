from PyQt5.QtCore import QDir

def create_project_structure(base_dir):
    base_qdir = QDir(base_dir)
    base_qdir.mkpath('normalizado/heStain')
    base_qdir.mkpath('train/print')
    base_qdir.mkpath('train/mitosis')
    base_qdir.mkpath('train/candidates')
    base_qdir.mkpath('test/mitosis')
    base_qdir.mkdir('anotations')

if __name__ == '__main__':
    create_project_structure('C:/Users/PelaoT/Desktop/Practica/dataset/mitos dataset')