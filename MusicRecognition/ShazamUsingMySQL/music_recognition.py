import os
from utils.print_utils import print_message

from core.STFTMusicProcessor import STFTMusicProcessorPredict
from database.MySQLConnector import MySQLConnector
from utils.hparam import hp


def predict_music(query_path):
    """
    对路径下的音频文件执行预测
    :param query_path:
    :return:
    """
    # 获取数据库的连接
    connector = MySQLConnector()
    # 获取核心的预测处理器
    music_processor = STFTMusicProcessorPredict()

    # 逐个预测数据
    pre_ret_map = dict()
    if os.path.isdir(query_path):
        query_file_list = os.listdir(query_path)
    elif os.path.isfile(query_path):
        query_file_list = [query_path]
    elif isinstance(query_path, list):
        query_file_list = query_path
    else:
        raise TypeError(f"{query_path} is not proper query, plz check again!")
    for path in query_file_list:
        if not (path.endswith("wav") or path.endswith("mp3")):
            continue
        # 获取音乐的相对路径
        # music_path = os.path.join(hp.fingerprint.path.query_path, path)
        print(f"Processing: {path}")
        # 预测歌曲
        music_info = music_processor.predict_music(music_path=path, connector=connector)
        # 根据music_info输出
        music_id = music_info['music_id']
        music_name = connector.find_music_name_by_music_id(music_id)
        print_message("预测歌曲：" + str(music_name) +
                      ", --- 线性匹配的Hash个数为：" + str(music_info['max_hash_count']) +
                      ", --- 歌曲偏移：" + str(music_info['music_offset']))
        pre_ret_map[path] = dict()
        pre_ret_map[path]["music_id"] = music_id
        pre_ret_map[path]["music_name"] = music_name
        pre_ret_map[path]["max_hash_count"] = music_info['max_hash_count']
        pre_ret_map[path]["music_offset"] = music_info['music_offset']
    try:
        connector.cursor.close()
        connector.conn.close()
    except Exception as e:
        print("Release mysql connection failed! ", e)
    return pre_ret_map


if __name__ == '__main__':
    predict_music("dataset/query/music_query/三频3--We Are Never Ever Getting Back Together.wav")
    # predict_music("dataset/query/music_query/中频4--渡口 女声.wav")
