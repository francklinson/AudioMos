import os
from utils.hparam import hp
from utils.print_utils import print_message
from database.MySQLConnector import MySQLConnector, DatabaseChecker
from core.STFTMusicProcessor import STFTMusicProcessorCreate


def add_music_fp_to_database():
    """
    在数据库中新增music footprint条目
    :return:
    """
    # 检查数据库
    dc = DatabaseChecker()
    dc.check_database()
    dc.check_tables()
    # 获取数据库的连接
    connector = MySQLConnector()
    # 短时傅里叶变化的处理
    music_processor = STFTMusicProcessorCreate()
    # 获取到歌曲的路径
    for path in os.listdir(hp.fingerprint.path.music_path):
        # 获得歌曲的路径
        music_path = os.path.join(hp.fingerprint.path.music_path, path)
        # 创建指纹并保存到数据库中
        music_processor.create_finger_prints_and_save_database(
            music_path=music_path,
            connector=connector)
        print_message(f"数据库中歌曲的条目 {path} 创建完成！！！")


if __name__ == '__main__':
    add_music_fp_to_database()
