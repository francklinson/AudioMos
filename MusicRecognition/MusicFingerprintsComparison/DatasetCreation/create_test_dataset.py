# 导入必要的库
import os  # 用于操作系统相关功能，如文件和目录操作
import csv  # 用于处理CSV文件
import time  # 用于时间相关操作，如延时
import traceback

import requests  # 用于发送HTTP请求
import random  # 用于生成随机数
from tqdm import tqdm  # 用于显示进度条


class JamendoTestDownloader:
    def __init__(self, client_id, data_dir='TEST_METADATA'):
        """
        初始化下载器类
        :param client_id: Jamendo API的客户端ID
        :param data_dir: 存储数据的目录
        """
        self.client_id = client_id
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, 'downloaded_tracks')  # 音频文件存储目录
        self.input_csv = os.path.join(data_dir, 'metadata_1000.csv')  # 输入CSV文件路径
        self.output_csv = os.path.join(data_dir, 'metadata_with_genre.csv')  # 输出CSV文件路径
        self.api_url = 'https://api.jamendo.com/v3.0/tracks'  # Jamendo API的URL
        self.max_songs = 200  # 最大下载数量
        self.id_range = (1, 10000)  # 歌曲ID的随机范围

        # 创建必要的目录
        os.makedirs(self.data_dir, exist_ok=True)  # 创建数据目录
        os.makedirs(self.audio_dir, exist_ok=True)  # 创建音频目录

    def load_existing_ids(self):
        """加载已存在的歌曲ID"""
        if os.path.exists(self.input_csv):
            with open(self.input_csv, newline='', encoding='utf-8') as f:
                return {row['track_id'] for row in csv.DictReader(f)}
        return set()

    def setup_csv_writer(self):
        """设置CSV写入器"""
        fieldnames = ['track_id', 'title', 'artist', 'album', 'duration_sec',
                      'download_url', 'filename', 'genre']
        self.csv_file = open(self.output_csv, 'w', newline='', encoding='utf-8')
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

    def get_track_info(self, track_id):
        """获取指定ID的歌曲信息"""
        params = {
            'client_id': self.client_id,
            'id': track_id,
            'format': 'json',
            'include': 'musicinfo',
            'limit': 1,
        }
        try:
            resp = requests.get(self.api_url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json().get('results', [])
        except Exception as e:
            print(f'Error fetching track info for {track_id}: {e}\n{traceback.format_exc()}')
            return []

    def download_audio(self, metadata):
        """下载音频文件"""
        download_url = metadata['download_url']
        try:
            audio_resp = requests.get(download_url, stream=True, timeout=20)
            audio_resp.raise_for_status()
            file_path = os.path.join(self.audio_dir, metadata['filename'])
            with open(file_path, 'wb') as af:
                for chunk in audio_resp.iter_content(1024):
                    af.write(chunk)
            return True
        except Exception as e:
            print(f'Error downloading audio from {download_url}: {e}\n{traceback.format_exc()}')
            return False

    def process_track(self, track_id):
        """处理单个音轨"""
        results = self.get_track_info(track_id)
        if not results:
            return False

        entry = results[0]
        music_info = entry.get('musicinfo', {})
        genres = music_info.get('tags', {}).get('genres', [])

        if not genres:
            return False

        metadata = {
            'track_id': entry['id'],
            'title': entry.get('name', ''),
            'artist': entry.get('artist_name', ''),
            'album': entry.get('album_name', ''),
            'duration_sec': entry.get('duration', ''),
            'download_url': entry.get('audiodownload', ''),
            'filename': f"{entry['id']}.mp3",
            'genre': genres[0]
        }

        if self.download_audio(metadata):
            self.writer.writerow(metadata)
            return True
        return False

    def run(self):
        """运行完整的下载流程"""
        existing_ids = self.load_existing_ids()
        self.setup_csv_writer()

        downloaded = 0
        checked_ids = set()

        with tqdm(total=self.max_songs, desc="Downloading test tracks") as pbar:
            while downloaded < self.max_songs:
                track_id = str(random.randint(*self.id_range))
                if track_id in existing_ids or track_id in checked_ids:
                    continue

                checked_ids.add(track_id)
                if self.process_track(track_id):
                    downloaded += 1
                    pbar.update(1)
                    pbar.set_postfix({'track_id': track_id})

                time.sleep(1.0)

        self.csv_file.close()
        print(
            f"\nDone. Retrieved {downloaded} new tracks. Metadata saved to {self.output_csv} and audio files in {self.audio_dir}")


if __name__ == "__main__":
    CLIENT_ID = os.getenv('JAMENDO_CLIENT_ID')
    downloader = JamendoTestDownloader(CLIENT_ID)
    downloader.run()
