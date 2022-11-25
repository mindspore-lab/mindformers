'''download_tools'''
import time
import requests
import urllib3
from tqdm import tqdm

from .logger import logger
class StatusCode:
    '''StatusCode'''
    succeed = 200


def downlond_with_progress_bar(url, filepath, chunk_size=1024, timeout=4):
    '''downlond_with_progress_bar'''
    start = time.time()

    try:
        response = requests.get(url, stream=True, timeout=timeout)
    except (TimeoutError, urllib3.exceptions.MaxRetryError,
            requests.exceptions.ProxyError) as exc:
        raise ConnectionError(f"Connect error, please download {url} to {filepath}.") from exc

    size = 0
    content_size = int(response.headers['content-length'])
    if response.status_code == StatusCode.succeed:
        logger.info('Start download %s,[File size]:{%.2f} MB',
                    filepath, content_size / chunk_size /1024)
        with open(filepath, 'wb') as file:
            for data in tqdm(response.iter_content(chunk_size=chunk_size)):
                file.write(data)
                size += len(data)
        file.close()
        end = time.time()
        logger.info('Download completed!,times: %.2fs', (end - start))
    else:
        raise KeyError(f"{url} is unconnected!")
