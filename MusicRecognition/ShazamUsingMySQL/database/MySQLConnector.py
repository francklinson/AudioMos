import pymysql
from utils.hparam import hp
from utils.print_utils import print_error, print_message

import abc


class IConnector(abc.ABC):

    def __init__(self):
        pass

    # 连接的方法
    @abc.abstractmethod
    def _connection(self):
        raise NotImplementedError(u"出错了，你没有实现_connection抽象方法")

    # 存储指纹
    @abc.abstractmethod
    def store_finger_prints(self, hashes, music_id_fk):
        raise NotImplementedError(u"出错了，你没有实现store_finger_prints抽象方法")

    # 保存一个指纹的方法
    @abc.abstractmethod
    def _add_finger_print(self, item, music_id_fk):
        raise NotImplementedError(u"出错了，你没有实现add_finger_print抽象方法")

    # 根据音乐的路径查找音乐
    @abc.abstractmethod
    def find_music_by_music_path(self, music_path):
        raise NotImplementedError(u"出错了，你没有实现find_music_by_music_path抽象方法")

    # 根据音乐id查找这首歌曲有多少Hash个数
    @abc.abstractmethod
    def calculation_hash_num_by_music_id(self, music_id):
        raise NotImplementedError(u"出错了，你没有实现calculation_hash_num_by_music_id抽象方法")

    # 添加歌曲
    @abc.abstractmethod
    def add_music(self, music_path):
        raise NotImplementedError(u"出错了，你没有实现add_music抽象方法")

    # 查找一个指纹
    @abc.abstractmethod
    def _find_finger_print(self, hash):
        raise NotImplementedError(u"出错了，你没有实现_find_finger_print抽象方法")

    # 查找指纹
    @abc.abstractmethod
    def find_math_hash(self, hashes):
        raise NotImplementedError(u"出错了，你没有实现find_math_hash抽象方法")


class MySQLConnector(IConnector):
    def __init__(self):
        # 调用父类的构造方法
        super().__init__()
        # 获取的数据库连接
        self.conn = None
        # 初始化游标
        self.cursor = None
        # 连接数据库，赋值游标和数据库连接
        self._connection()

    # 连接的方法
    def _connection(self):

        # 获取到mysql的连接
        self.conn = pymysql.connect(
            # 主机名
            host=hp.fingerprint.database.host,
            # 端口号
            port=hp.fingerprint.database.port,
            # 用户名
            user=hp.fingerprint.database.user,
            # 用户密码
            password=hp.fingerprint.database.password,
            # 数据库名称
            database=hp.fingerprint.database.database,
            # 字符集
            charset=hp.fingerprint.database.charset,
        )

        # 游标
        self.cursor = self.conn.cursor()

    # 保存一个指纹的方法
    def _add_finger_print(self, item, music_id_fk):

        # SQL
        sql = "insert into %s(%s, %s, %s) values(%s, '%s', '%s')" % (
            hp.fingerprint.database.tables.finger_prints.name,
            hp.fingerprint.database.tables.finger_prints.column.music_id_fk,
            hp.fingerprint.database.tables.finger_prints.column.hash,
            hp.fingerprint.database.tables.finger_prints.column.offset,
            music_id_fk,
            item[0],
            item[1]
        )

        # 执行SQL语句
        self.cursor.execute(sql)
        self.conn.commit()

    # 存储指纹
    def store_finger_prints(self, hashes, music_id_fk):
        # 遍历指纹，一个一个保存到数据库（hash,offset）
        for item in hashes:
            self._add_finger_print(item=item, music_id_fk=music_id_fk)

    # 根据音乐的路径查找音乐
    def find_music_by_music_path(self, music_path):
        # SQL
        sql = "select %s from %s where %s = '%s'" % (
            # 列名
            hp.fingerprint.database.tables.music.column.music_id,
            # 表名
            hp.fingerprint.database.tables.music.name,
            # 列名
            hp.fingerprint.database.tables.music.column.music_path,
            # 传入的参数
            music_path
        )
        # 执行SQL
        self.cursor.execute(sql)
        # 拿到返回值
        result = self.cursor.fetchone()
        if result is None:
            return None
        else:
            return result[0]

    def find_music_name_by_music_id(self, music_id):
        """
        根据歌曲id查询歌曲名称
        :param music_id:
        :return:
        """
        sql = "select %s from %s where %s = '%s'" % (
            # 列名
            hp.fingerprint.database.tables.music.column.music_name,
            # 表名
            hp.fingerprint.database.tables.music.name,
            # 列名
            hp.fingerprint.database.tables.music.column.music_id,
            # 传入的参数
            music_id
        )
        # 执行SQL
        self.cursor.execute(sql)
        # 拿到返回值
        result = self.cursor.fetchone()
        if result is None:
            return None
        else:
            return result[0]

    # 根据音乐id查找这首歌曲有多少Hash个数
    def calculation_hash_num_by_music_id(self, music_id):
        # SQL
        sql = "select count('%s') from %s where %s = %s" % (
            hp.fingerprint.database.tables.finger_prints.column.id_fp,
            hp.fingerprint.database.tables.finger_prints.name,
            hp.fingerprint.database.tables.finger_prints.column.music_id_fk,
            music_id
        )

        # 执行SQL
        self.cursor.execute(sql)

        # 拿到返回值
        result = self.cursor.fetchone()
        if result is None:
            return 0
        else:
            return result[0]

    # 添加歌曲
    def add_music(self, music_path):

        # SQL
        sql = "insert into %s(%s, %s) values ('%s', '%s')" % (
            hp.fingerprint.database.tables.music.name,
            hp.fingerprint.database.tables.music.column.music_name,
            hp.fingerprint.database.tables.music.column.music_path,
            music_path.split(hp.fingerprint.path.split)[-1],
            music_path
        )

        # 执行SQL
        self.cursor.execute(sql)
        self.conn.commit()

        # 根据music_path查找歌曲id
        music_id = self.find_music_by_music_path(music_path=music_path)
        return music_id

    # 查找一个指纹
    def _find_finger_print(self, hash_):
        # SQL
        sql = "select %s, %s from %s where %s = '%s'" % (
            hp.fingerprint.database.tables.finger_prints.column.music_id_fk,
            hp.fingerprint.database.tables.finger_prints.column.offset,
            hp.fingerprint.database.tables.finger_prints.name,
            hp.fingerprint.database.tables.finger_prints.column.hash,
            hash_
        )
        # 执行SQL
        self.cursor.execute(sql)
        # 拿到返回值
        result = self.cursor.fetchone()
        return result

    # 查找指纹
    # @cost_time
    def find_math_hash_old(self, hashes):
        # 一个一个的查找指纹，[hash, offset]
        for item in hashes:
            ret = self._find_finger_print(item[0])
            if ret is not None:
                music_id_fk, offset_database = ret
            else:
                continue
            # 待查指纹对应数据库中的歌曲id，待查指纹在数据库中的偏移，待查指纹在待查音乐片段中的偏移
            yield music_id_fk, offset_database, item[1]

    # 查找指纹
    # @cost_time
    def find_math_hash(self, hashes):
        # 一个一个的查找指纹，[hash, offset]
        mapper = {}
        for audio_hash, offset in hashes:
            mapper[audio_hash] = offset

        hash_list = []
        for item in hashes:
            hash_, t1 = item[0], item[1]
            hash_list.append(hash_)

        sql = "select %s, %s, %s from %s where %s in %s" % (
            hp.fingerprint.database.tables.finger_prints.column.hash,
            hp.fingerprint.database.tables.finger_prints.column.music_id_fk,
            hp.fingerprint.database.tables.finger_prints.column.offset,
            hp.fingerprint.database.tables.finger_prints.name,
            hp.fingerprint.database.tables.finger_prints.column.hash,
            tuple(hash_list)
        )
        # 执行SQL
        self.cursor.execute(sql)
        # 拿到返回值
        result = self.cursor.fetchall()
        # print(result)

        for audio_hash, sid, offset in result:
            # offset和 mapper[audio_hash]
            yield sid, offset, mapper[audio_hash]

        # for item in hashes:
        #     ret = self._find_finger_print(item[0])
        #     if ret is not None:
        #         music_id_fk, offset_database = ret
        #     else:
        #         continue
        #     # 待查指纹对应数据库中的歌曲id，待查指纹在数据库中的偏移，待查指纹在待查音乐片段中的偏移
        #     yield music_id_fk, offset_database, item[1]


class DatabaseChecker:
    def __init__(self):
        """
        -- 创建名为 finger_prints 的数据库，并设置字符集为 utf8
        CREATE DATABASE IF NOT EXISTS `finger_prints` CHARACTER SET utf8;

        -- 创建 music 表
        CREATE TABLE IF NOT EXISTS `music` (
            `music_id` INT AUTO_INCREMENT PRIMARY KEY,
            `music_name` VARCHAR(255) NOT NULL,
            `music_path` VARCHAR(255) NOT NULL
        );

        -- 创建 finger_prints 表，并设置外键关联到 music 表的 music_id 字段
        CREATE TABLE IF NOT EXISTS `finger_prints` (
            `id_fp` INT AUTO_INCREMENT PRIMARY KEY,
            `music_id_fk` INT NOT NULL,
            `hash` VARCHAR(255) NOT NULL,
            `offset` INT NOT NULL,
            FOREIGN KEY (`music_id_fk`) REFERENCES `music`(`music_id`)
        );
        """
        self.database_host = hp.fingerprint.database.host
        self.database_user = hp.fingerprint.database.user
        self.database_password = hp.fingerprint.database.password
        self.database_name = hp.fingerprint.database.database
        self.database_charset = 'utf8mb4'

    def check_database(self):
        """
        检查数据库是否正常，若不存在则新建数据库
        :return:
        """
        # 数据库连接配置
        config = {
            'host': self.database_host,  # 数据库主机地址
            'user': self.database_user,  # 数据库用户名
            'password': self.database_password,  # 数据库密码
            'charset': self.database_charset
        }
        cursor, connection = None, None
        database_name = hp.fingerprint.database.database
        try:
            # 建立数据库连接
            connection = pymysql.connect(**config)
            # 创建游标对象
            cursor = connection.cursor()
            # 定义要创建的数据库名称
            # 执行创建数据库的SQL语句
            create_database_query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
            cursor.execute(create_database_query)
            print_message(f"数据库 {database_name} 状态正常！")
        except pymysql.Error as e:
            print_error(f"创建数据库  {database_name} 时发生错误: {e}")
        finally:
            # 关闭游标和连接
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()

    def check_tables(self):
        """
        检查数据库中的表单是否正常
        :return:
        """
        # 数据库连接配置
        config = {
            'host': self.database_host,  # 数据库主机地址
            'user': self.database_user,  # 数据库用户名
            'password': self.database_password,  # 数据库密码
            'database': self.database_name,
            'charset': self.database_charset
        }
        cursor, connection = None, None
        try:
            # 建立数据库连接
            connection = pymysql.connect(**config)
            # 创建游标对象
            cursor = connection.cursor()
            # 执行创建table的SQL语句
            music_table_name = hp.fingerprint.database.tables.music.name
            music_id = hp.fingerprint.database.tables.music.column.music_id
            music_name = hp.fingerprint.database.tables.music.column.music_name
            music_path = hp.fingerprint.database.tables.music.column.music_path

            fp_table_name = hp.fingerprint.database.tables.finger_prints.name
            id_fp = hp.fingerprint.database.tables.finger_prints.column.id_fp
            music_id_fk = hp.fingerprint.database.tables.finger_prints.column.music_id_fk
            hash_ = hp.fingerprint.database.tables.finger_prints.column.hash
            offset = hp.fingerprint.database.tables.finger_prints.column.offset
            # 查询、创建music table
            create_music_table_query = f"CREATE TABLE IF NOT EXISTS  {music_table_name} ( {music_id} INT AUTO_INCREMENT PRIMARY KEY,{music_name} VARCHAR(255) NOT NULL,{music_path} VARCHAR(255) NOT NULL)"
            cursor.execute(create_music_table_query)
            # 查询、创建fp table
            create_fp_table_query = f"CREATE TABLE IF NOT EXISTS {fp_table_name} ({id_fp} INT AUTO_INCREMENT PRIMARY KEY,{music_id_fk} INT NOT NULL, {hash_} VARCHAR(255) NOT NULL,{offset} INT NOT NULL,FOREIGN KEY ({music_id_fk}) REFERENCES {music_table_name}({music_id}))"
            cursor.execute(create_fp_table_query)

        except Exception as e:
            print_error(e)
