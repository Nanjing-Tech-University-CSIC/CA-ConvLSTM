import logging


def get_logger(fname=None, silent=False):
    if fname is None:
        logging.basicConfig(
            level=logging.INFO if not silent else logging.CRITICAL,
            format='%(name)-12s: %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filemode='w'
        )
    else:
        logging.basicConfig(
            level=logging.INFO if not silent else logging.CRITICAL,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=fname,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    # 创建log日志
    log_name = 'sst-202205012.log'
    get_logger(fname='sst-202205012.log')
    logging.info('Experiment log path in: {}'.format(log_name))