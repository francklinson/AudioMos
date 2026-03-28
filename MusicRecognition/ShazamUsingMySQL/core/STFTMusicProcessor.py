import matplotlib.pyplot as plt

from core.IMusicProcessor import IMusicProcessorCreate, IMusicProcessorPredict
from utils.data_utils import ProcessTimer
from utils.hparam import hp
from utils.print_utils import print_message, print_warning


class STFTMusicProcessorCreate(IMusicProcessorCreate):

    # 创建指纹并且保存到数据库中
    def create_finger_prints_and_save_database(self, music_path, connector):
        """
        创建指纹并且保存到数据库中
        :param music_path: 音乐路径
        :param connector: 连接数据库的
        :return: 无
        """
        # 先查询看看数据库中是否有这首歌
        music_id = connector.find_music_by_music_path(music_path=music_path)

        # 如果数据库中没有这首歌
        if music_id is None:
            # 添加歌曲，拿到歌曲id
            music_id = connector.add_music(music_path)

            # 计算Hash
            hashes = list(self._calculation_hash(music_path=music_path))

            # 将Hash值保存到数据库中
            connector.store_finger_prints(hashes=hashes, music_id_fk=music_id)

            # 歌曲的Hash个数
            hash_num = connector.calculation_hash_num_by_music_id(music_id=music_id)
            # 打印提示信息
            print_message("歌曲：" + str(music_id) + " 添加成功! \nHash数目为：" + str(hash_num) + "\n")
        # 如果数据库中有这首歌
        else:
            # 计算这首歌曲的Hash个数
            hash_num = connector.calculation_hash_num_by_music_id(music_id=music_id)
            print_warning("这首歌曲 " + str(music_id) + " 已经存在，一共有" + str(hash_num) + "条Hash!!!")


class STFTMusicProcessorPredict(IMusicProcessorPredict):

    # 预测歌曲
    def predict_music(self, music_path, connector):
        """
        计算Hash
        :param music_path:
        :param connector:
        :return:
        """
        hash_ = list(self._calculation_hash(music_path=music_path))

        # 看是否开启了显示时间
        start = None
        p = ProcessTimer()
        if hp.fingerprint.show_time:
            p.start_time()

        # 根据Hash在数据库中查找，[hash, offset]
        match_hash_list = set(connector.find_math_hash(hashes=hash_))

        if hp.fingerprint.show_plot.predict_plot.hash_plot:
            self._show_line_plot(match_hash_list)

        # 看是否开启了显示时间
        if hp.fingerprint.show_time:
            p.end_time('在数据库中查找花费的时间')

        return self._align_match(match_hash_list=match_hash_list)

    # @cost_time
    @staticmethod
    def _align_match(match_hash_list):
        """
        待查指纹对应数据库中的歌曲id，待查指纹在数据库中的偏移，待查指纹在待查音乐片段中的偏移
        :param match_hash_list:
        :return:
        """
        # 最终返回的歌曲id
        music_id = -1
        # 最终返回的歌曲的偏移
        music_offset = -1
        # 最终返回的查找到匹配的Hash个数
        max_hash_count = -1

        # 保存返回的结果
        result = {}

        for matches in match_hash_list:
            # 拿到音乐的id, 数据库中的偏移，查询片段自身的偏移
            music_id_fk, offset_database, offset_query = matches

            # 计算数据库中音乐的偏移和查询片段自身的偏移之间的差值
            offset = int(int(offset_database) - int(offset_query))

            # 如果offset不存在字典里，则添加进去
            if offset not in result:
                result[offset] = {}

            if music_id_fk not in result[offset]:
                result[offset][music_id_fk] = 0

            # 统计在当前偏移下，歌曲的出现次数
            result[offset][music_id_fk] += 1

            if result[offset][music_id_fk] > max_hash_count:
                # 更新max_hash_count的值
                max_hash_count = result[offset][music_id_fk]
                # 赋值歌曲id
                music_id = music_id_fk
                # 赋值歌曲的offset
                music_offset = offset
        return {
            "music_id": music_id,
            "music_offset": music_offset,
            "max_hash_count": max_hash_count
        }

    @staticmethod
    def _show_line_plot(match_hash):
        """
        绘制线性关系图
        :param match_hash:
        :return:
        """
        c = [item[0] for item in match_hash]

        x_and_y = [(item[1], item[2]) for item in match_hash]

        x = [int(item[0]) for item in x_and_y]
        y = [int(item[0]) for item in x_and_y]

        plt.scatter(x, y, c=c, marker='o')
        plt.show()
