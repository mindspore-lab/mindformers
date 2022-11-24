import time
import requests

from .logger import logger

class StatusCode:
    succeed = 200


def downlond_with_progress_bar(url, filepath, chunk_size = 1024):
    start = time.time()
    response = requests.get(url, stream=True)
    size = 0
    content_size = int(response.headers['content-length'])

    if response.status_code == StatusCode.succeed:
        logger.info('Start download {filepath},[File size]:{size:.2f} MB'.format(filepath=filepath, size = content_size / chunk_size /1024))
        with open(filepath,'wb') as file:
            for data in response.iter_content(chunk_size = chunk_size):
                file.write(data)
                size +=len(data)
                logger.info('\r'+'[Process]:%s%.2f%%' % ('>'*int(size*50/ content_size), float(size / content_size * 100)))
        file.close()
        end = time.time()
        logger.info('Download completed!,times: %.2fs' % (end - start))
    else:
        raise KeyError(f"{url} is unconnected!")