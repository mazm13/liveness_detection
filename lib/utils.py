import os

dic = {'client': 'ClientRaw', 'imposter': 'ImposterRaw'}

def read_path(content_file):
    prefix = dic[content_file.split('_')[0]]
    f = open(os.path.join('./raw', content_file), 'r')
    paths = []
    for line in f:
        path = os.path.join('./raw', prefix, line)
        path = path.replace('\\', '/')
        path = path.strip()
        paths.append(path)
    f.close()
    return paths
