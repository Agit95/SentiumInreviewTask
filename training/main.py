import os
if __name__ != '__main__':
    print('ERROR  "{}" file is a main file for London house prices prediction.'.format(os.path.join(os.getcwd(), 'main.py')))
    exit(0)

from application import CApplication

application = CApplication()
application.run()