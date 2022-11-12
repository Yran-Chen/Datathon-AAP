import os
import configparser
import logging
import time

root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'..')

class LogHandler():
    def __init__(self,name = 'test',setLvl = logging.NOTSET):
        try:
            tmp_dir = self.get_log_route()
        except configparser.NoSectionError:
            try:
                os.listdir(os.path.join(root_dir, 'Logfile'))
            except FileNotFoundError:
                os.mkdir(os.path.join(root_dir, 'Logfile'))
            finally:
                tmp_dir = root_dir

        logging.root.setLevel(setLvl)
        log_file = os.path.join(tmp_dir, 'Logfile', '{}'.format(name))
        _log = logging.getLogger()
        f_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_handler.setFormatter(formatter)
        _log.addHandler(f_handler)
        _log.setLevel(logging.INFO)
        self._log = _log

    def get_log_route(self):
        conf_file = os.path.join(root_dir, "conf", 'application.conf')
        conf = configparser.ConfigParser()
        conf.read(conf_file)
        return conf.get('log', 'route')

def kwargDecorator(decorator):

    def decorator_proxy(f=None, **kwargs):
        if f is not None:
            return decorator(f=f, **kwargs)

        def decorator_proxy(func):
            return decorator(f=f, **kwargs)

        return decorator_proxy

    return decorator_proxy

def log(_log):

    def decorator(f):

        def wrapper(ins, *args,**kwargs):
            _log.info('{} start at {}'.format(f.__name__, str(time.ctime())))
            _log.info('********************************************************************************')
            _log.info('input json:')
            for k, v in ins.param.items():
                _log.info('{}, {}'.format(k, v))
            _log.info('********************************************************************************')
            try:
                a = f(ins, *args,**kwargs)
                _log.info('{} end at {}'.format(f.__name__, str(time.ctime())))
            except Exception as e:
                _log.error(e, exc_info=True)
                raise e
            finally:
                _log.info('********************************************************************************')
                _log.info('********************************************************************************')

        return wrapper

    return decorator

def logTime(_log):

    def decorator(f):

        def wrapper(ins, **kwargs):
            _log.info('********************************************************************************')

            _log.info('{} start at {}'.format(f.__name__, str(time.ctime())))
            start_time = time.time()
            try:
                a = f(ins, **kwargs)
                _log.info('{} end at {}'.format(f.__name__, str(time.ctime())))
                end_time = time.time()

                _log.info('Time usage is: {}'.format( str(end_time - start_time) )  )
            except Exception as e:
                _log.error(e, exc_info=True)
                raise e
            finally:
                return a
                _log.info('********************************************************************************')

        return wrapper

    return decorator

import codecs
import json
import errno
import pickle



def _save_pickle(file, file_dir):
        fn = os.path.abspath(file_dir)
        create_dir(fn)
        with open(fn, 'wb') as f:
            pickle.dump(file, f)
        f.close()


def _read_pickle(fp):
        content = dict()
        try:
            with open(fp, 'rb') as f:
                content = pickle.load(f)
        except IOError as e:
            if e.errno not in (errno.ENOENT, errno.EISDIR, errno.EINVAL):
                raise
        return content


def _read_json(fp):
        content = dict()
        try:
            with codecs.open(fp, 'r', encoding='utf-8') as f:
                content = json.load(f)
        except IOError as e:
            if e.errno not in (errno.ENOENT, errno.EISDIR, errno.EINVAL):
                raise
        return content


def _save_json(serializable, file_dir):
        fn = os.path.abspath(file_dir)
        create_dir(fn)
        with codecs.open(fn, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, separators=(',\n', ': '))


def create_dir(file_dir):
        if not os.path.exists(os.path.dirname(file_dir)):
            try:
                os.makedirs(os.path.dirname(file_dir))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise EnvironmentError
                else:
                    print("???")