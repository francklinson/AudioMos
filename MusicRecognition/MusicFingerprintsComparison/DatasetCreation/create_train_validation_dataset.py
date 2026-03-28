import os
import random
import requests
import pandas as pd
from tqdm import tqdm
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3, TIT2, TPE1, TALB


class JamendoDownloader:
    def __init__(self, client_id, audio_dir="audio_1000"):
        self.client_id = client_id
        self.audio_dir = audio_dir
        self.metadata_csv = os.path.join(audio_dir, "metadata.csv")
        self.failed_log = os.path.join(audio_dir, "failed_ids.txt")
        os.makedirs(audio_dir, exist_ok=True)

    def fetch_tracks(self, target_count=5000):
        """从Jamendo API获取音轨信息"""
        pool, page, per_page = [], 1, 200
        while len(pool) < target_count:
            resp = requests.get(
                "https://api.jamendo.com/v3.0/tracks/",
                params={
                    "client_id": self.client_id,
                    "format": "json",
                    "limit": per_page,
                    "offset": (page - 1) * per_page,
                    "audioformat": "mp32",
                    "filter": "audiodownload_allowed=1",
                    "include": "musicinfo"
                },
            )
            resp.raise_for_status()
            data = resp.json().get("results", [])
            if not data:
                break
            pool.extend(data)
            page += 1
            print(f"-> Collected {len(pool)} tracks so far...")
        return pool

    def create_metadata(self):
        """创建或加载元数据CSV文件"""
        if not os.path.exists(self.metadata_csv):
            print("Getting tracks from Jamendo API...")
            pool = self.fetch_tracks()
            if len(pool) < 1000:
                raise RuntimeError("Insufficient tracks available")
            # 从5000个备选中随机抽取1000个音轨
            sampled = random.sample(pool, 1000)
            md = pd.DataFrame(sampled)[[
                "id", "name", "artist_name", "album_name",
                "duration", "audiodownload"
            ]]
            md = md.rename(columns={
                "id": "track_id",
                "name": "title",
                "artist_name": "artist",
                "album_name": "album",
                "duration": "duration_sec",
                "audiodownload": "download_url"
            })
            md["filename"] = md["track_id"] + ".mp3"
            md.to_csv(self.metadata_csv, index=False)
            print(f"metadata.csv created with {len(md)} tracks.")
            return md
        else:
            print("metadata.csv found, resuming downloads.")
            md = pd.read_csv(self.metadata_csv, dtype={"track_id": str})
            md["filename"] = md["track_id"] + ".mp3"
            return md

    def set_id3_tags(self, file_path, title, artist, album):
        """设置MP3文件的ID3标签"""
        try:
            audio = EasyID3(file_path)
        except:
            id3 = ID3(file_path)
            id3.add(TIT2(encoding=3, text=title))
            id3.add(TPE1(encoding=3, text=artist))
            id3.add(TALB(encoding=3, text=album))
            id3.save()
            audio = EasyID3(file_path)

        audio["title"] = title
        audio["artist"] = artist
        audio["album"] = album
        audio.save()

    def download_tracks(self, metadata):
        """下载音轨文件"""
        failed = []
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Downloading MP3s"):
            out_path = os.path.join(self.audio_dir, row["filename"])
            if os.path.exists(out_path):
                continue
            try:
                resp = requests.get(row["download_url"], stream=True, timeout=30)
                resp.raise_for_status()
                with open(out_path, "wb") as wf:
                    for chunk in resp.iter_content(8192):
                        wf.write(chunk)

                self.set_id3_tags(out_path, row["title"], row["artist"], row["album"])
            except Exception as e:
                failed.append(row["track_id"])
                tqdm.write(f"Failed download {row['track_id']}: {e}")

        if failed:
            with open(self.failed_log, "w") as f:
                f.write("\n".join(failed))
            print(f"\n{len(failed)} tracks failed to download. See {self.failed_log}")

    def validate_downloads(self, metadata):
        """验证下载的文件"""
        mismatches, corrupted, missing = [], [], []
        print("\nValidating downloaded MP3 durations...")

        for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
            file_path = os.path.join(self.audio_dir, row["filename"])
            if not os.path.exists(file_path):
                missing.append(row["filename"])
                continue

            try:
                audio = MP3(file_path)
                actual_duration = audio.info.length
                expected_duration = row["duration_sec"]
                if abs(actual_duration - expected_duration) > 5:
                    mismatches.append({
                        "filename": row["filename"],
                        "expected": expected_duration,
                        "actual": round(actual_duration, 2)
                    })
            except Exception as e:
                corrupted.append((row["filename"], str(e)))

        self.print_validation_results(missing, corrupted, mismatches)

    def print_validation_results(self, missing, corrupted, mismatches):
        """打印验证结果"""
        print("\nSummary:")

        if missing:
            print(f"\nMissing files ({len(missing)}):")
            for f in missing:
                print(f"  - {f}")

        if corrupted:
            print(f"\nCorrupted files ({len(corrupted)}):")
            for fname, err in corrupted:
                print(f"  - {fname}: {err}")

        if mismatches:
            print(f"\nDuration mismatches ({len(mismatches)}):")
            for m in mismatches:
                print(f"  - {m['filename']}: expected {m['expected']}s, actual {m['actual']}s")

        if not (missing or corrupted or mismatches):
            print("All 1000 tracks downloaded successfully")
        else:
            total_issues = len(missing) + len(corrupted) + len(mismatches)
            print(f"\nCompleted with {total_issues} issues.")

    def run(self):
        """运行完整的下载和验证流程"""
        metadata = self.create_metadata()
        self.download_tracks(metadata)
        self.validate_downloads(metadata)
        print("\ncompleted.")


if __name__ == "__main__":
    CLIENT_ID = os.getenv('JAMENDO_CLIENT_ID')
    downloader = JamendoDownloader(CLIENT_ID)
    downloader.run()
