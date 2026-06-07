"""
降噪测评实验结果数据库

使用SQLite进行轻量级实验记录持久化，零额外依赖。
支持实验的存储、查询、对比和导出。

数据库Schema:
- experiments: 实验基本信息
- results: 测评结果数据 (JSON)
- algorithm_scores: 各算法各指标得分
- metrics_history: 指标历史趋势
"""

import os
import json
import time
import sqlite3
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class ExperimentSummary:
    """实验摘要"""
    experiment_id: str
    name: str
    description: str = ""
    dataset: str = ""
    n_algorithms: int = 0
    n_files: int = 0
    status: str = "unknown"
    created_at: str = ""
    duration_seconds: float = 0.0
    git_commit: str = ""
    tags: List[str] = field(default_factory=list)
    best_algorithms: Dict[str, str] = field(default_factory=dict)  # metric->algo


class ExperimentDB:
    """
    实验结果SQLite数据库

    使用方式:
        db = ExperimentDB("./data/experiments/results.db")
        db.save_experiment(experiment_result)
        experiments = db.list_experiments(dataset="dns_challenge")
        db.compare_experiments(ids)
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "./data/experiments/results.db"):
        """
        初始化数据库

        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """初始化数据库表"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    config_json TEXT DEFAULT '{}',
                    dataset TEXT DEFAULT '',
                    n_algorithms INTEGER DEFAULT 0,
                    n_files INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    duration_seconds REAL DEFAULT 0,
                    git_commit TEXT DEFAULT '',
                    tags TEXT DEFAULT '[]',
                    best_algorithms TEXT DEFAULT '{}',
                    benchmark_result_json TEXT DEFAULT '{}',
                    statistical_report_json TEXT DEFAULT '{}',
                    chart_files_json TEXT DEFAULT '[]',
                    metadata_json TEXT DEFAULT '{}'
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS algorithm_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    algorithm_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    mean_value REAL,
                    std_value REAL,
                    median_value REAL,
                    min_value REAL,
                    max_value REAL,
                    n_samples INTEGER DEFAULT 0,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
                    UNIQUE(experiment_id, algorithm_name, metric_name)
                )
            """)

            # 索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_created
                ON experiments(created_at DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_dataset
                ON experiments(dataset)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scores_exp_algo
                ON algorithm_scores(experiment_id, algorithm_name)
            """)

            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")

    @contextmanager
    def _get_conn(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ===========================
    # CRUD 操作
    # ===========================

    def save_experiment(self, experiment_result: Any) -> str:
        """
        保存实验结果

        Args:
            experiment_result: 可以是 ExperimentResult 对象或包含实验数据的 dict

        Returns:
            experiment_id
        """
        # 支持 dict 和 ExperimentResult 对象
        if hasattr(experiment_result, 'experiment_id'):
            exp_id = experiment_result.experiment_id
            name = getattr(experiment_result.config, 'name', 'Unnamed')
            desc = getattr(experiment_result.config, 'description', '')
            dataset = getattr(getattr(experiment_result.config, 'dataset', None), 'name', '')
            n_algos = len(getattr(getattr(experiment_result, 'benchmark_result', None), 'algorithms', {}))
            n_files = getattr(getattr(experiment_result, 'benchmark_result', None), 'dataset_info', {}).get('n_files', 0)
            created = getattr(experiment_result, 'created_at', datetime.now().isoformat())
            duration = getattr(experiment_result, 'duration', 0)
            git_commit = getattr(experiment_result, 'git_commit_hash', '')
        elif isinstance(experiment_result, dict):
            exp_id = experiment_result.get('experiment_id', f"exp_{int(time.time())}")
            name = experiment_result.get('name', 'Unnamed')
            desc = experiment_result.get('description', '')
            dataset = experiment_result.get('dataset', '')
            n_algos = experiment_result.get('n_algorithms', 0)
            n_files = experiment_result.get('n_files', 0)
            created = experiment_result.get('created_at', datetime.now().isoformat())
            duration = experiment_result.get('duration', 0)
            git_commit = experiment_result.get('git_commit', '')
        else:
            exp_id = f"exp_{int(time.time())}"
            name = str(experiment_result)
            desc = ''
            dataset = ''
            n_algos = 0
            n_files = 0
            created = datetime.now().isoformat()
            duration = 0
            git_commit = ''

        now = datetime.now().isoformat()

        with self._get_conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments
                (experiment_id, name, description, config_json, dataset,
                 n_algorithms, n_files, status, created_at, updated_at,
                 duration_seconds, git_commit)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?)
            """, (exp_id, name, desc, '{}', dataset,
                  n_algos, n_files, created, now, duration, git_commit))

        logger.info(f"实验已保存: {exp_id}")
        return exp_id

    def save_algorithm_scores(
        self, experiment_id: str, algorithm_name: str,
        metrics: Dict[str, Dict[str, float]],
    ):
        """保存算法指标得分"""
        with self._get_conn() as conn:
            for metric_name, stats in metrics.items():
                conn.execute("""
                    INSERT OR REPLACE INTO algorithm_scores
                    (experiment_id, algorithm_name, metric_name,
                     mean_value, std_value, median_value, min_value, max_value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id, algorithm_name, metric_name,
                    stats.get("mean"), stats.get("std"), stats.get("median"),
                    stats.get("min"), stats.get("max"),
                ))

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """获取单个实验详情"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchone()

            if not row:
                return None

            result = dict(row)
            # 解析JSON字段
            for field in ['config_json', 'tags', 'best_algorithms',
                         'benchmark_result_json', 'statistical_report_json',
                         'chart_files_json', 'metadata_json']:
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except json.JSONDecodeError:
                        pass

            # 加载得分
            scores = conn.execute(
                "SELECT * FROM algorithm_scores WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchall()
            result['scores'] = [dict(s) for s in scores]

            return result

    def list_experiments(
        self,
        dataset: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[ExperimentSummary]:
        """列出实验"""
        conditions = []
        params = []

        if dataset:
            conditions.append("dataset = ?")
            params.append(dataset)
        if status:
            conditions.append("status = ?")
            params.append(status)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT experiment_id, name, description, dataset,
                   n_algorithms, n_files, status, created_at,
                   duration_seconds, git_commit, tags, best_algorithms
            FROM experiments
            {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        summaries = []
        for row in rows:
            d = dict(row)
            if d.get('tags') and isinstance(d['tags'], str):
                try:
                    d['tags'] = json.loads(d['tags'])
                except json.JSONDecodeError:
                    d['tags'] = []
            if d.get('best_algorithms') and isinstance(d['best_algorithms'], str):
                try:
                    d['best_algorithms'] = json.loads(d['best_algorithms'])
                except json.JSONDecodeError:
                    d['best_algorithms'] = {}
            summaries.append(ExperimentSummary(**d))

        return summaries

    def delete_experiment(self, experiment_id: str) -> bool:
        """删除实验及其关联数据"""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM algorithm_scores WHERE experiment_id = ?", (experiment_id,))
            conn.execute("DELETE FROM experiments WHERE experiment_id = ?", (experiment_id,))
        logger.info(f"实验已删除: {experiment_id}")
        return True

    # ===========================
    # 查询和对比
    # ===========================

    def compare_experiments(
        self, experiment_ids: List[str], metrics: Optional[List[str]] = None
    ) -> Dict:
        """
        对比多个实验

        Args:
            experiment_ids: 实验ID列表
            metrics: 关注的指标列表

        Returns:
            对比结果字典
        """
        if metrics is None:
            metrics = ["pesq", "stoi", "sisdr", "dnsmos_ovrl"]

        comparison = {
            "experiments": [],
            "metrics_comparison": {},
        }

        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp:
                comparison["experiments"].append({
                    "id": exp["experiment_id"],
                    "name": exp["name"],
                    "dataset": exp.get("dataset", ""),
                    "created_at": exp.get("created_at", ""),
                })

        # 收集各实验的得分
        for metric in metrics:
            metric_data = {}
            with self._get_conn() as conn:
                for exp_id in experiment_ids:
                    rows = conn.execute(
                        """SELECT algorithm_name, mean_value
                           FROM algorithm_scores
                           WHERE experiment_id = ? AND metric_name = ?""",
                        (exp_id, metric)
                    ).fetchall()

                    metric_data[exp_id] = {
                        row["algorithm_name"]: row["mean_value"]
                        for row in rows
                    }

            comparison["metrics_comparison"][metric] = metric_data

        return comparison

    def get_algorithm_history(
        self, algorithm_name: str, metric: Optional[str] = None
    ) -> List[Dict]:
        """
        获取同一算法的历史趋势

        Args:
            algorithm_name: 算法名称
            metric: 关注的指标

        Returns:
            历史数据点列表
        """
        query = """
            SELECT e.experiment_id, e.name, e.created_at, e.dataset,
                   s.metric_name, s.mean_value, s.std_value
            FROM algorithm_scores s
            JOIN experiments e ON s.experiment_id = e.experiment_id
            WHERE s.algorithm_name = ?
        """
        params = [algorithm_name]

        if metric:
            query += " AND s.metric_name = ?"
            params.append(metric)

        query += " ORDER BY e.created_at ASC"

        with self._get_conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(r) for r in rows]

    def find_best_overall(self, metric: str = "pesq", limit: int = 3) -> List[Dict]:
        """
        查找历史上某指标最佳的算法

        Args:
            metric: 指标名称
            limit: 返回数量

        Returns:
            最佳算法列表
        """
        query = """
            SELECT algorithm_name, MAX(mean_value) as best_score,
                   COUNT(*) as experiment_count
            FROM algorithm_scores
            WHERE metric_name = ?
            GROUP BY algorithm_name
            ORDER BY best_score DESC
            LIMIT ?
        """
        with self._get_conn() as conn:
            rows = conn.execute(query, (metric, limit)).fetchall()

        return [dict(r) for r in rows]

    def get_statistics(self) -> Dict:
        """获取数据库统计信息"""
        with self._get_conn() as conn:
            total_exps = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            total_algo_scores = conn.execute("SELECT COUNT(DISTINCT algorithm_name) FROM algorithm_scores").fetchone()[0]
            total_metrics = conn.execute("SELECT COUNT(DISTINCT metric_name) FROM algorithm_scores").fetchone()[0]

            return {
                "db_path": self.db_path,
                "db_size_mb": round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if os.path.exists(self.db_path) else 0,
                "total_experiments": total_exps,
                "total_algorithms_tracked": total_algo_scores,
                "total_metrics_tracked": total_metrics,
            }

    def export_experiment(self, experiment_id: str, output_path: str) -> bool:
        """导出实验数据为JSON"""
        exp = self.get_experiment(experiment_id)
        if not exp:
            logger.error(f"实验不存在: {experiment_id}")
            return False

        # 转换非可序列化类型
        def convert(obj):
            if isinstance(obj, bytes):
                return obj.decode('utf-8', errors='replace')
            return str(obj)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(exp, f, indent=2, ensure_ascii=False, default=convert)

        logger.info(f"实验已导出: {output_path}")
        return True
